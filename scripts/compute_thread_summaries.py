"""Batch-computes and persists per-ticket thread summaries (SPEC M6: "apply
to multi-message tickets"; accept criterion: "2-line summary shown at top of
every ticket view"). Same precompute-and-persist shape as
scripts/compute_sentiment_trajectories.py (M5): the dashboard reads
task="thread_summary" Predictions via GET /tickets/{id}/predictions, it never
calls POST /predict/summary live -- summarization's CPU latency budget (SPEC
§3: up to 3s/request) is far slower than entities' live-per-page-load
call, so precomputing keeps ticket-page loads fast regardless of model
latency (see docs/decisions.md).

Only tickets with >= 2 messages get a summary -- SPEC's own phrase is
"multi-message tickets"; a single-message ticket has nothing to summarize
and the dashboard's best-effort fetch just renders nothing for it, the same
graceful-degrade pattern the sparkline already follows for tickets with no
sentiment_trajectory Prediction yet.

Predictor: ExtractiveSummaryPredictor (lead-k) by default. Pass --model
transformer once ml/training/train_summarization.py's FLAN-T5-small config
has actually been run on a GPU and docs/m6-comparison-report.md confirms
it's worth deploying (see docs/m6-how-to-run-locally.md) -- until then the
transformer export at models/transformer_thread_summary_flan-t5-small_v1/final
is only the CPU smoke-test artifact from that config's --max-steps 5 run,
not a real model.

Full-recompute semantics, scoped to the tickets actually processed in this
run (not every task="thread_summary" row globally, unlike M5's trajectory
backfill): every run deletes existing Predictions only for the tickets it's
about to reinsert, in the same transaction as the inserts -- Prediction ids
are random UUIDs, not deterministic, so re-running isn't naturally a no-op
(same reasoning as M5's backfill script, see docs/decisions.md). Scoping the
delete (rather than wiping the whole task) matters once --limit is used: a
capped run reprocesses a subset without discarding every other ticket's
already-computed summary.

--limit caps how many tickets get (re)processed, oldest-ingested first --
useful for the transformer path, where CPU seq2seq generation is far slower
than the baseline (~0.4s/ticket batched, so the full ~63k-ticket corpus is a
multi-hour run; a few hundred is enough to demo the feature and feed the
LLM-judge sample without blocking on the full corpus).

Run:
  uv run python scripts/compute_thread_summaries.py
  uv run python scripts/compute_thread_summaries.py --model transformer --limit 500
"""

import argparse
from pathlib import Path

from api.db.models import AuthorRole, Prediction, Ticket
from api.db.session import SessionLocal
from sqlalchemy import delete, select
from sqlalchemy.orm import Session, selectinload

from ml.inference.base import SummaryPredictor, format_dialogue
from ml.inference.extractive_summary import DEFAULT_K, ExtractiveSummaryPredictor

SUMMARY_TRANSFORMER_DIR = Path("models/transformer_thread_summary_flan-t5-small_v1/final")
BASELINE_MODEL_VERSION = "baseline_thread_summary_v1"
TRANSFORMER_MODEL_VERSION = "transformer_thread_summary_flan-t5-small_v1"

BATCH_SIZE = 16  # smaller than M5's 64: seq2seq generation is far more expensive per text
TASK = "thread_summary"
MIN_MESSAGES = 2

_SPEAKER_LABEL = {AuthorRole.CUSTOMER: "Customer", AuthorRole.AGENT: "Agent"}


def _get_summary_predictor(model: str) -> tuple[SummaryPredictor, str]:
    if model == "transformer":
        from ml.inference.summarization import SummarizationPredictor

        return SummarizationPredictor(SUMMARY_TRANSFORMER_DIR), TRANSFORMER_MODEL_VERSION
    return ExtractiveSummaryPredictor(), BASELINE_MODEL_VERSION


def _predict_in_batches(predictor: SummaryPredictor, texts: list[str]) -> list[str]:
    summaries: list[str] = []
    for start in range(0, len(texts), BATCH_SIZE):
        summaries.extend(r.summary for r in predictor.predict(texts[start : start + BATCH_SIZE]))
    return summaries


def compute_and_persist(session: Session, tickets: list[Ticket], model: str) -> int:
    tickets = [t for t in tickets if len(t.messages) >= MIN_MESSAGES]
    if not tickets:
        return 0

    predictor, model_version = _get_summary_predictor(model)

    dialogues = [
        format_dialogue([(_SPEAKER_LABEL[m.author_role], m.text_clean) for m in t.messages])
        for t in tickets
    ]
    summaries = _predict_in_batches(predictor, dialogues)

    ticket_ids = [t.id for t in tickets]
    session.execute(
        delete(Prediction).where(Prediction.task == TASK, Prediction.ticket_id.in_(ticket_ids))
    )

    for ticket, summary in zip(tickets, summaries, strict=True):
        session.add(
            Prediction(
                ticket_id=ticket.id,
                task=TASK,
                label=summary,
                payload={"message_count": len(ticket.messages)},
                model_version=model_version,
            )
        )

    session.commit()
    return len(tickets)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["baseline", "transformer"], default="baseline")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap how many tickets to (re)process, oldest-ingested first (default: all)",
    )
    args = parser.parse_args()

    session = SessionLocal()
    try:
        stmt = select(Ticket).options(selectinload(Ticket.messages)).order_by(Ticket.ingested_at)
        if args.limit is not None:
            stmt = stmt.limit(args.limit)
        tickets = list(session.scalars(stmt).all())
        print(
            f"computing {TASK} for {len(tickets)} tickets (model={args.model}, "
            f"baseline k={DEFAULT_K}, min_messages={MIN_MESSAGES})"
        )
        written = compute_and_persist(session, tickets, args.model)
        print(f"wrote {written} {TASK} predictions")
    finally:
        session.close()


if __name__ == "__main__":
    main()
