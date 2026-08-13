"""Loads the git-committed demo seed (data/seed/*.jsonl, produced once by
scripts/export_demo_seed.py against the real dev corpus) into a freshly
migrated database -- the M10 zero-setup demo path (`make demo` ->
`docker compose exec api python -m ml.data.seed_demo`).

Unlike ml/data/cli.py's `seed` subcommand (an 18-row test-fixture seed used
by unit tests, untouched by this module), this is the real "few hundred
curated tickets, every capability populated" demo dataset SPEC M10 asks
for: tickets + messages, their already-computed sentiment_trajectory/
thread_summary/topic Predictions, the full Topic catalog, all 40 KB
articles, and the current eval_runs table -- plus a Chroma index rebuild
(resolved tickets + KB articles) so search/RAG work immediately too.

Every insert here is idempotent (ON CONFLICT DO NOTHING on stable,
pre-assigned ids from the committed JSONL, or upsert for the Chroma step) --
safe to re-run `make demo` against an already-seeded database.

Run (inside the api container, or locally against a dev DB):
  uv run python -m ml.data.seed_demo
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from api.db.models import EvalRun, KbArticle, Prediction, Topic
from api.db.session import SessionLocal
from api.schemas.ticket import CanonicalTicket
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from ml.data.persist import persist_tickets

SEED_DIR = Path(__file__).resolve().parents[2] / "data" / "seed"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _uuid_or_none(value: str | None) -> uuid.UUID | None:
    return uuid.UUID(value) if value is not None else None


def _dt_or_none(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value is not None else None


def seed_tickets(session: Session) -> int:
    rows = _read_jsonl(SEED_DIR / "tickets.jsonl")
    tickets = (CanonicalTicket(**row) for row in rows)
    return persist_tickets(session, tickets)


def seed_predictions(session: Session) -> int:
    rows = _read_jsonl(SEED_DIR / "predictions.jsonl")
    if not rows:
        return 0
    values = [
        {
            "id": uuid.UUID(r["id"]),
            "ticket_id": _uuid_or_none(r["ticket_id"]),
            "message_id": _uuid_or_none(r["message_id"]),
            "task": r["task"],
            "label": r["label"],
            "score": r["score"],
            "payload": r["payload"],
            "model_version": r["model_version"],
            "eval_run_id": _uuid_or_none(r["eval_run_id"]),
            "created_at": _dt_or_none(r["created_at"]),
        }
        for r in rows
    ]
    stmt = pg_insert(Prediction).values(values).on_conflict_do_nothing(index_elements=["id"])
    session.execute(stmt)
    session.commit()
    return len(values)


def seed_topics(session: Session) -> int:
    rows = _read_jsonl(SEED_DIR / "topics.jsonl")
    if not rows:
        return 0
    values = [
        {
            "id": uuid.UUID(r["id"]),
            "topic_key": r["topic_key"],
            "label": r["label"],
            "keywords": r["keywords"],
            "size": r["size"],
            "model_version": r["model_version"],
            "created_at": _dt_or_none(r["created_at"]),
        }
        for r in rows
    ]
    stmt = pg_insert(Topic).values(values).on_conflict_do_nothing(index_elements=["id"])
    session.execute(stmt)
    session.commit()
    return len(values)


def seed_kb_articles(session: Session) -> int:
    rows = _read_jsonl(SEED_DIR / "kb_articles.jsonl")
    if not rows:
        return 0
    values = [
        {
            "id": uuid.UUID(r["id"]),
            "title": r["title"],
            "body": r["body"],
            "tags": r["tags"],
            "source_kind": r["source_kind"],
            "source_key": r["source_key"],
            "generator_version": r["generator_version"],
            "created_at": _dt_or_none(r["created_at"]),
        }
        for r in rows
    ]
    stmt = pg_insert(KbArticle).values(values).on_conflict_do_nothing(index_elements=["id"])
    session.execute(stmt)
    session.commit()
    return len(values)


def seed_eval_runs(session: Session) -> int:
    rows = _read_jsonl(SEED_DIR / "eval_runs.jsonl")
    if not rows:
        return 0
    values = [
        {
            "id": uuid.UUID(r["id"]),
            "task": r["task"],
            "model_version": r["model_version"],
            "dataset": r["dataset"],
            "split": r["split"],
            "metrics": r["metrics"],
            "params": r["params"],
            "git_sha": r["git_sha"],
            "status": r["status"],
            "started_at": _dt_or_none(r["started_at"]),
            "finished_at": _dt_or_none(r["finished_at"]),
        }
        for r in rows
    ]
    stmt = pg_insert(EvalRun).values(values).on_conflict_do_nothing(index_elements=["id"])
    session.execute(stmt)
    session.commit()
    return len(values)


def index_chroma(session: Session) -> None:
    """Rebuilds the two M8 Chroma collections from whatever's now in
    Postgres (the just-seeded resolved tickets + all KB articles) -- reuses
    scripts/index_search_corpus.py's own functions rather than
    reimplementing them; both this module and that script ship in the API
    image (infra/api.Dockerfile), so the cross-import is safe there. Lazy
    import: sentence-transformers/chromadb live behind the `search`
    dependency group, not necessarily present on a bare host `make install`
    venv -- only paid at demo-seed time, same reasoning
    scripts/index_search_corpus.py's own module docstring gives."""
    from ml.inference.vector_store import ChromaVectorStore
    from scripts.index_search_corpus import (
        index_kb_articles,
        index_resolved_tickets,
        load_resolved_tickets,
    )

    store = ChromaVectorStore()
    tickets = load_resolved_tickets(session)
    n_tickets = index_resolved_tickets(store, tickets)
    print(f"indexed {n_tickets} resolved tickets into Chroma")

    articles = list(session.scalars(select(KbArticle)).all())
    n_articles = index_kb_articles(store, articles)
    print(f"indexed {n_articles} KB articles into Chroma")


def main() -> None:
    from api.config import get_settings

    session = SessionLocal()
    try:
        print(f"seeded {seed_tickets(session)} tickets")
        print(f"seeded {seed_predictions(session)} predictions")
        print(f"seeded {seed_topics(session)} topics")
        print(f"seeded {seed_kb_articles(session)} kb articles")
        print(f"seeded {seed_eval_runs(session)} eval runs")
        # Skipped entirely when SEARCH_ENABLED=false -- not just pointless
        # (nothing would ever query the index) but actively what causes
        # apps/api/config.py's documented 512MB overrun on boot: this is
        # what forces the sentence-transformers/torch import on every
        # container start, regardless of whether any request ever needs it.
        if get_settings().search_enabled:
            index_chroma(session)
        else:
            print("search_enabled=false -- skipping Chroma reindex")
    finally:
        session.close()


if __name__ == "__main__":
    main()
