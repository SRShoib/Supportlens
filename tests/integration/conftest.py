import contextlib
from collections.abc import Iterator
from typing import TYPE_CHECKING

import pytest
from api.db.session import make_engine
from sqlalchemy import text
from sqlalchemy.orm import Session, sessionmaker
from testcontainers.community.postgres import PostgresContainer

if TYPE_CHECKING:
    from testcontainers.community.chroma import ChromaContainer

    from ml.inference.vector_store import ChromaVectorStore


def _docker_available() -> bool:
    try:
        import docker

        docker.from_env().ping()
    except Exception:
        return False
    return True


DOCKER_AVAILABLE = _docker_available()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if DOCKER_AVAILABLE:
        return
    skip_marker = pytest.mark.skip(reason="Docker daemon not reachable; skipping integration tests")
    for item in items:
        item.add_marker(skip_marker)


@pytest.fixture(scope="session")
def postgres_container() -> Iterator[PostgresContainer]:
    with PostgresContainer("postgres:16-alpine") as pg:
        yield pg


@pytest.fixture(scope="session")
def database_url(postgres_container: PostgresContainer) -> str:
    return postgres_container.get_connection_url().replace("psycopg2", "psycopg")


@pytest.fixture
def migrated_db(database_url: str) -> str:
    """Ensures the schema is at head before a test runs. Function-scoped and
    idempotent (a no-op if already at head) so it's safe regardless of what
    test_migrations.py's own upgrade/downgrade cycle did before this test."""
    from alembic.config import Config

    from alembic import command

    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(config, "head")
    return database_url


@pytest.fixture
def db_session(migrated_db: str) -> Iterator[Session]:
    engine = make_engine(migrated_db)
    session = sessionmaker(bind=engine)()
    # The container is session-scoped for speed, so start every test from a
    # clean slate rather than accumulating rows across tests in this module.
    session.execute(
        text(
            "TRUNCATE TABLE predictions, messages, tickets, eval_runs, llm_calls, topics, "
            "kb_articles CASCADE"
        )
    )
    session.commit()
    yield session
    session.close()


@pytest.fixture(scope="session")
def chroma_container() -> Iterator["ChromaContainer"]:
    # chromadb lives behind the `search` dependency group (SPEC M8), not
    # synced by default -- same reason ml/inference/vector_store.py lazily
    # imports it. Skips this fixture (and every test that depends on it)
    # cleanly rather than failing when it's absent, e.g. CI's default
    # `--group serving` sync.
    pytest.importorskip("chromadb")
    from testcontainers.community.chroma import ChromaContainer

    # Pinned to match infra/docker-compose.yml's production service exactly
    # -- the testcontainers default (chromadb/chroma:1.0.0) speaks a
    # different heartbeat API version (v2) than what's actually deployed
    # (0.5.23, v1), so testing against the default would validate a server
    # this project doesn't run in production.
    with ChromaContainer(image="chromadb/chroma:0.5.23") as chroma:
        yield chroma


# The two collection names apps/api/routers/search.py and rag.py actually
# query (ml/inference/retrieval.py). Tests that need to exercise the real
# router against real Chroma have no choice but to use these exact names;
# tests that just want an isolated scratch collection should mint their own
# unique name instead (see test_vector_store_chroma.py's _collection_name)
# rather than relying on the reset below.
_ROUTER_COLLECTIONS = ("resolved_tickets", "kb_articles")


@pytest.fixture
def chroma_store(chroma_container: "ChromaContainer") -> "ChromaVectorStore":
    """A real ChromaVectorStore talking to the real containerized server --
    unlike every other M8 test's injected fake store, this exercises
    ml/inference/vector_store.py's actual chromadb.HttpClient wiring
    (CLAUDE.md: "Integration tests use testcontainers for... Chroma").
    Function-scoped: each test gets a fresh ChromaVectorStore instance, but
    the container itself is session-scoped for speed, so this resets the
    two well-known router collections before every test -- same "clean
    slate per test" contract db_session's TRUNCATE gives Postgres, just via
    delete-and-recreate since Chroma has no TRUNCATE equivalent."""
    import chromadb

    config = chroma_container.get_config()
    raw_client = chromadb.HttpClient(host=config["host"], port=config["port"])
    for name in _ROUTER_COLLECTIONS:
        with contextlib.suppress(Exception):
            raw_client.delete_collection(name)  # didn't exist yet -- also a clean slate

    from ml.inference.vector_store import ChromaVectorStore

    return ChromaVectorStore(host=config["host"], port=config["port"])
