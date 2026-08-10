from collections.abc import Iterator

import pytest
from testcontainers.community.postgres import PostgresContainer


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
    from alembic import command
    from alembic.config import Config

    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(config, "head")
    return database_url
