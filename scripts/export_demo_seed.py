"""Dev-only tool: curates a small, portable demo dataset from the real dev
Postgres (already populated by M1-M9's real runs) and writes it to
data/seed/*.jsonl -- small enough to commit directly to git (CLAUDE.md:
"committed data limited to small fixtures", same precedent as
data/gold/ner_gold_v1.jsonl), the source ml/data/seed_demo.py reads at
`make demo` time.

Never run in CI/tests; this is a one-off curation tool, re-run by hand
whenever the demo seed needs refreshing. Requires the real dev Postgres
(all of M1-M9's real runs already applied) -- not meant to run against a
fresh/empty database.

Curation targets (~300-400 tickets total, see docs/decisions.md):
- Bitext: up to N_PER_INTENT tickets per intent (27 intents) -- even
  coverage of the intent classifier's whole label space.
- Twitter: a general sample from the corpus's real dense high-volume weeks
  (DENSE_WEEKS -- the same window scripts/compute_drift.py uses), biased
  toward tickets that already have a persisted thread_summary and/or a
  positive resolution_quality (RAG-ready), topped up with a plain random
  sample. Plus a deliberate oversample of SPIKE_TOPIC_KEY in its real spike
  week (SPIKE_WEEK) and a handful of other weeks -- preserves one real
  SPEC M7 emerging-issue alarm (z-score > 2) at reduced scale rather than
  only shipping a demo where that panel is always empty.
- Every selected ticket's already-computed sentiment_trajectory/
  thread_summary/topic Predictions travel with it, plus the full Topic
  catalog, all 40 KB articles, and the current eval_runs table (eval
  numbers are corpus-wide facts, not tied to which tickets get seeded --
  safe to ship as-is).

Run: uv run python scripts/export_demo_seed.py
"""

import json
import random
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from api.db.models import EvalRun, KbArticle, Prediction, Ticket, TicketSource, Topic
from api.db.session import SessionLocal
from api.schemas.ticket import CanonicalTicket
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "seed"

SEED = 42
N_PER_INTENT = 5

# The Twitter corpus's real stable high-volume tail (see
# scripts/compute_drift.py's module docstring for how this was found --
# same window, reused here for consistency).
DENSE_WEEKS = (
    "2017-10-09",
    "2017-10-16",
    "2017-10-23",
    "2017-10-30",
    "2017-11-06",
    "2017-11-13",
    "2017-11-20",
    "2017-11-27",
)
N_RESOLVED = 30
N_SUMMARIZED = 30
N_RANDOM_FILL = 90

# topic_key=9 ("package, delivery, delivered, packages") really spikes on
# SPIKE_WEEK in the full corpus (z=3.09, count=89, share 3.4%) -- measured
# directly via apps/api/routers/topics.py::_compute_topic_volume against
# the real dev DB, see docs/decisions.md. Oversampling it here (relative to
# its true share) is what lets the emerging-issues panel actually fire in
# the much smaller seed rather than only ever showing its empty state.
SPIKE_TOPIC_KEY = 9
SPIKE_WEEK = "2017-11-27"
SPIKE_WEEK_COUNT = 30
HISTORY_WEEKS = ("2017-10-09", "2017-10-23", "2017-11-06", "2017-11-13")
HISTORY_WEEK_COUNT = 6

PREDICTION_TASKS = ("sentiment_trajectory", "thread_summary", "topic")


def _week_start(dt: datetime) -> str:
    monday = dt.date() - timedelta(days=dt.weekday())
    return monday.isoformat()


def _bitext_sample(session: Session) -> list[Ticket]:
    intents = session.scalars(
        select(Ticket.meta["intent"].astext).where(Ticket.source == TicketSource.BITEXT).distinct()
    ).all()

    selected: list[Ticket] = []
    for intent in sorted(intents):
        stmt = (
            select(Ticket)
            .options(selectinload(Ticket.messages))
            .where(Ticket.source == TicketSource.BITEXT, Ticket.meta["intent"].astext == intent)
            .order_by(Ticket.external_id)
            .limit(N_PER_INTENT)
        )
        selected.extend(session.scalars(stmt).all())
    return selected


def _twitter_dense_window_ids_by_week(session: Session) -> dict[str, list[str]]:
    rows = session.execute(
        select(Ticket.id, Ticket.created_at).where(
            Ticket.source == TicketSource.TWITTER, Ticket.created_at.isnot(None)
        )
    ).all()
    by_week: dict[str, list[str]] = defaultdict(list)
    for ticket_id, created_at in rows:
        week = _week_start(created_at)
        if week in DENSE_WEEKS:
            by_week[week].append(str(ticket_id))
    return by_week


def _topic_ticket_ids(session: Session, topic_key: int, week: str) -> list[str]:
    rows = session.execute(
        select(Ticket.id, Ticket.created_at, Prediction.payload)
        .join(Prediction, Prediction.ticket_id == Ticket.id)
        .where(
            Prediction.task == "topic",
            Ticket.source == TicketSource.TWITTER,
            Ticket.created_at.isnot(None),
        )
    ).all()
    return [
        str(ticket_id)
        for ticket_id, created_at, payload in rows
        if payload.get("topic_key") == topic_key and _week_start(created_at) == week
    ]


def _resolved_ticket_ids(session: Session, candidate_ids: set[str]) -> list[str]:
    rows = session.execute(
        select(Ticket.id)
        .join(Prediction, Prediction.ticket_id == Ticket.id)
        .where(Prediction.task == "sentiment_trajectory", Prediction.score > 0)
    ).all()
    return [str(r[0]) for r in rows if str(r[0]) in candidate_ids]


def _summarized_ticket_ids(session: Session, candidate_ids: set[str]) -> list[str]:
    rows = session.execute(
        select(Prediction.ticket_id).where(Prediction.task == "thread_summary")
    ).all()
    return [str(r[0]) for r in rows if str(r[0]) in candidate_ids]


def _twitter_sample(session: Session, rng: random.Random) -> list[Ticket]:
    by_week = _twitter_dense_window_ids_by_week(session)
    all_ids = {tid for ids in by_week.values() for tid in ids}

    selected_ids: set[str] = set()

    # Spike boost first -- these ids must survive the general sampling caps
    # below untouched.
    spike_ids = _topic_ticket_ids(session, SPIKE_TOPIC_KEY, SPIKE_WEEK)
    rng.shuffle(spike_ids)
    selected_ids.update(spike_ids[:SPIKE_WEEK_COUNT])

    for week in HISTORY_WEEKS:
        history_ids = _topic_ticket_ids(session, SPIKE_TOPIC_KEY, week)
        rng.shuffle(history_ids)
        selected_ids.update(history_ids[:HISTORY_WEEK_COUNT])

    remaining_pool = list(all_ids - selected_ids)

    resolved = _resolved_ticket_ids(session, set(remaining_pool))
    rng.shuffle(resolved)
    for tid in resolved[:N_RESOLVED]:
        selected_ids.add(tid)

    remaining_pool = list(all_ids - selected_ids)
    summarized = _summarized_ticket_ids(session, set(remaining_pool))
    rng.shuffle(summarized)
    for tid in summarized[:N_SUMMARIZED]:
        selected_ids.add(tid)

    remaining_pool = list(all_ids - selected_ids)
    rng.shuffle(remaining_pool)
    selected_ids.update(remaining_pool[:N_RANDOM_FILL])

    stmt = (
        select(Ticket)
        .options(selectinload(Ticket.messages))
        .where(Ticket.id.in_([uuid.UUID(i) for i in selected_ids]))
    )
    return list(session.scalars(stmt).all())


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows)} rows -> {path}")


def _export_tickets(tickets: list[Ticket]) -> None:
    rows = [
        json.loads(CanonicalTicket.model_validate(t, from_attributes=True).model_dump_json())
        for t in tickets
    ]
    _write_jsonl(OUTPUT_DIR / "tickets.jsonl", rows)


def _export_predictions(session: Session, ticket_ids: set[str]) -> None:
    stmt = select(Prediction).where(
        Prediction.task.in_(PREDICTION_TASKS),
        Prediction.ticket_id.in_([uuid.UUID(i) for i in ticket_ids]),
    )
    predictions = session.scalars(stmt).all()
    rows = [
        {
            "id": str(p.id),
            "ticket_id": str(p.ticket_id) if p.ticket_id else None,
            "message_id": str(p.message_id) if p.message_id else None,
            "task": p.task,
            "label": p.label,
            "score": p.score,
            "payload": p.payload,
            "model_version": p.model_version,
            "eval_run_id": None,  # never referenced by these three tasks
            "created_at": p.created_at.isoformat(),
        }
        for p in predictions
    ]
    _write_jsonl(OUTPUT_DIR / "predictions.jsonl", rows)


def _export_topics(session: Session) -> None:
    topics = session.scalars(select(Topic)).all()
    rows = [
        {
            "id": str(t.id),
            "topic_key": t.topic_key,
            "label": t.label,
            "keywords": t.keywords,
            "size": t.size,
            "model_version": t.model_version,
            "created_at": t.created_at.isoformat(),
        }
        for t in topics
    ]
    _write_jsonl(OUTPUT_DIR / "topics.jsonl", rows)


def _export_kb_articles(session: Session) -> None:
    articles = session.scalars(select(KbArticle)).all()
    rows = [
        {
            "id": str(a.id),
            "title": a.title,
            "body": a.body,
            "tags": a.tags,
            "source_kind": a.source_kind,
            "source_key": a.source_key,
            "generator_version": a.generator_version,
            "created_at": a.created_at.isoformat(),
        }
        for a in articles
    ]
    _write_jsonl(OUTPUT_DIR / "kb_articles.jsonl", rows)


def _export_eval_runs(session: Session) -> None:
    runs = session.scalars(select(EvalRun)).all()
    rows = [
        {
            "id": str(r.id),
            "task": r.task,
            "model_version": r.model_version,
            "dataset": r.dataset,
            "split": r.split,
            "metrics": r.metrics,
            "params": r.params,
            "git_sha": r.git_sha,
            "status": r.status,
            "started_at": r.started_at.isoformat(),
            "finished_at": r.finished_at.isoformat() if r.finished_at else None,
        }
        for r in runs
    ]
    _write_jsonl(OUTPUT_DIR / "eval_runs.jsonl", rows)


def main() -> None:
    rng = random.Random(SEED)
    session = SessionLocal()
    try:
        bitext_tickets = _bitext_sample(session)
        twitter_tickets = _twitter_sample(session, rng)
        tickets = bitext_tickets + twitter_tickets
        ticket_ids = {str(t.id) for t in tickets}
        print(f"selected {len(bitext_tickets)} bitext + {len(twitter_tickets)} twitter tickets")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        _export_tickets(tickets)
        _export_predictions(session, ticket_ids)
        _export_topics(session)
        _export_kb_articles(session)
        _export_eval_runs(session)
    finally:
        session.close()


if __name__ == "__main__":
    main()
