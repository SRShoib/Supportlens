"""Batch-computes and persists per-ticket sentiment trajectories (SPEC M5:
"aggregate per-ticket into a trajectory... per-ticket aggregate stored as a
Prediction"). Unlike M2-M4's live, stateless /predict/* endpoints, this is
the first module that needs durably-stored predictions -- the dashboard's
sparkline (GET /tickets/{id}/predictions?task=sentiment_trajectory) reads
what this script writes, it never recomputes live.

Sentiment predictor: BaselinePredictor by default. Pass --model transformer
once ml/training/train_transformer.py's sentiment config has actually been
run on a GPU and docs/m5-comparison-report.md confirms it's worth deploying
(see docs/m5-how-to-run-locally.md) -- until then the transformer export is
only the CPU smoke-test artifact from that config's --max-steps 5 run, not a
real model.

Urgency predictor: always the M3-established winner
(transformer_urgency_deberta-v3-small_v1, docs/m3-comparison-report.md's
+0.1151 macro-F1 win over baseline), reused as-is rather than re-decided
here -- see ml/inference/sentiment_trajectory.py's module docstring for why
"ticket urgency" means the urgency prediction on the ticket's first customer
message specifically.

Full-recompute semantics: every run deletes existing
task="sentiment_trajectory" Prediction rows before reinserting, in the same
transaction as the inserts -- Prediction ids are random UUIDs, not
deterministic like Ticket/Message, so re-running isn't naturally a no-op the
way M1's loaders are (see docs/decisions.md).

Run:
  uv run python scripts/compute_sentiment_trajectories.py
  uv run python scripts/compute_sentiment_trajectories.py --model transformer
"""

import argparse
from pathlib import Path

from api.db.models import AuthorRole, Prediction, Ticket
from api.db.session import SessionLocal
from sqlalchemy import delete, select
from sqlalchemy.orm import Session, selectinload

from ml.inference.base import Predictor, TaskResult
from ml.inference.baseline import BaselinePredictor
from ml.inference.sentiment_trajectory import build_trajectory
from ml.inference.transformer import TransformerPredictor

SENTIMENT_BASELINE_PATH = Path("models/baseline_sentiment_v1/model.joblib")
SENTIMENT_TRANSFORMER_DIR = Path("models/transformer_sentiment_distilbert-base-uncased_v1/final")
URGENCY_TRANSFORMER_DIR = Path("models/transformer_urgency_deberta-v3-small_v1/final")
URGENCY_MODEL_VERSION = "transformer_urgency_deberta-v3-small_v1"

BATCH_SIZE = 64
TASK = "sentiment_trajectory"


def _get_sentiment_predictor(model: str) -> tuple[Predictor[TaskResult], str]:
    if model == "transformer":
        return TransformerPredictor(SENTIMENT_TRANSFORMER_DIR), (
            "transformer_sentiment_distilbert-base-uncased_v1"
        )
    return BaselinePredictor(SENTIMENT_BASELINE_PATH), "baseline_sentiment_v1"


def _predict_in_batches(predictor: Predictor[TaskResult], texts: list[str]) -> list[TaskResult]:
    results: list[TaskResult] = []
    for start in range(0, len(texts), BATCH_SIZE):
        results.extend(predictor.predict(texts[start : start + BATCH_SIZE]))
    return results


def _first_customer_text(ticket: Ticket) -> str:
    """The message ml/training/splits.py::build_urgency_splits trained the
    urgency model on the shape of -- the ticket's opening customer message,
    not whichever message happens to be first overall."""
    first_customer = next(
        (m for m in ticket.messages if m.author_role == AuthorRole.CUSTOMER), None
    )
    return (first_customer or ticket.messages[0]).text_clean


def compute_and_persist(session: Session, tickets: list[Ticket], model: str) -> int:
    tickets = [t for t in tickets if t.messages]
    if not tickets:
        return 0

    sentiment_predictor, sentiment_model_version = _get_sentiment_predictor(model)
    urgency_predictor = TransformerPredictor(URGENCY_TRANSFORMER_DIR)

    all_texts = [m.text_clean for t in tickets for m in t.messages]
    all_sentiments = _predict_in_batches(sentiment_predictor, all_texts)

    urgency_results = _predict_in_batches(
        urgency_predictor, [_first_customer_text(t) for t in tickets]
    )

    session.execute(delete(Prediction).where(Prediction.task == TASK))

    offset = 0
    for ticket, urgency_result in zip(tickets, urgency_results, strict=True):
        n = len(ticket.messages)
        sentiment_results = all_sentiments[offset : offset + n]
        offset += n

        is_customer = [m.author_role == AuthorRole.CUSTOMER for m in ticket.messages]
        trajectory = build_trajectory(sentiment_results, is_customer, urgency_result.label)

        session.add(
            Prediction(
                ticket_id=ticket.id,
                task=TASK,
                label=trajectory.final_customer_label,
                score=trajectory.resolution_quality,
                payload=trajectory.to_payload()
                | {
                    "urgency_label": urgency_result.label,
                    "urgency_score": urgency_result.score,
                    "urgency_model_version": URGENCY_MODEL_VERSION,
                },
                model_version=sentiment_model_version,
            )
        )

    session.commit()
    return len(tickets)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["baseline", "transformer"], default="baseline")
    args = parser.parse_args()

    session = SessionLocal()
    try:
        tickets = list(session.scalars(select(Ticket).options(selectinload(Ticket.messages))).all())
        print(f"computing {TASK} for {len(tickets)} tickets (sentiment model={args.model})")
        written = compute_and_persist(session, tickets, args.model)
        print(f"wrote {written} {TASK} predictions")
    finally:
        session.close()


if __name__ == "__main__":
    main()
