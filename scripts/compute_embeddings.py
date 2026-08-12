"""Batch-embeds every Twitter ticket's customer messages with
sentence-transformers (SPEC M7: "Embed the real-ticket corpus"), and writes
the result to a local artifact under data/embeddings/ -- not Chroma. SPEC
M8 is the module that introduces the Chroma client and application code
around it; M7 stays Postgres/local-artifact-only (docs/decisions.md).

Embedding unit: the concatenation of a ticket's CUSTOMER messages only
(text_clean, chronological order), not the full agent+customer thread.
Agent replies are template-heavy ("sorry to hear that, please DM us") and
would otherwise dominate every topic's terms; the customer's own words are
what actually describe the issue -- same instinct as
ml/inference/sentiment_trajectory.py's "final customer message" and
ml/training/splits.py's urgency split, both of which train on customer
text specifically. Tickets with no customer message at all (rare) are
skipped, not fallen back to the full thread -- unlike the sentiment
trajectory's edge case, a topic model has no obvious "whole thread" analog
that stays on-topic for a ticket that never speaks as the customer.

Scope: TicketSource.TWITTER only. Bitext is synthetic single-turn intent
utterances with created_at always NULL, so it can never appear on the
weekly trend axis SPEC M7 also asks for -- see docs/decisions.md.

Output (index-aligned by row order -- row i in one is ticket i in the
other):
  data/embeddings/tickets_minilm_v1.npy      float32 array, shape (n, dim)
  data/embeddings/tickets_minilm_v1.parquet  columns: ticket_id, document

Run:
  uv run python scripts/compute_embeddings.py
  uv run python scripts/compute_embeddings.py --limit 500   # smoke test
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from api.db.models import Ticket, TicketSource
from api.db.session import SessionLocal
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from ml.data.documents import build_documents

OUTPUT_DIR = Path("data/embeddings")
OUTPUT_STEM = "tickets_minilm_v1"
BATCH_SIZE = 64


def compute_and_write(session: Session, tickets: list[Ticket]) -> int:
    # Lazy import: sentence-transformers lives behind the `topics`
    # dependency group (not synced by default or in CI, see pyproject.toml)
    # -- keeping it out of this module's top-level imports means
    # ml.data.documents.build_documents stays importable/unit-testable
    # without that group installed, same pattern apps/api/routers/predict.py
    # uses for transformers/torch.
    from ml.inference.embeddings import SentenceEmbeddingPredictor

    ticket_ids, documents = build_documents(tickets)
    skipped = len(tickets) - len(documents)
    print(f"embedding {len(documents)} tickets ({skipped} skipped, no customer message)")
    if not documents:
        return 0

    predictor = SentenceEmbeddingPredictor()
    vectors = predictor.encode(documents, batch_size=BATCH_SIZE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / f"{OUTPUT_STEM}.npy", vectors.astype(np.float32))
    pd.DataFrame({"ticket_id": ticket_ids, "document": documents}).to_parquet(
        OUTPUT_DIR / f"{OUTPUT_STEM}.parquet"
    )
    return len(documents)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap how many tickets to embed, oldest-ingested first (default: all)",
    )
    args = parser.parse_args()

    session = SessionLocal()
    try:
        stmt = (
            select(Ticket)
            .options(selectinload(Ticket.messages))
            .where(Ticket.source == TicketSource.TWITTER)
            .order_by(Ticket.ingested_at)
        )
        if args.limit is not None:
            stmt = stmt.limit(args.limit)
        tickets = list(session.scalars(stmt).all())
        written = compute_and_write(session, tickets)
        print(f"wrote {written} embeddings to {OUTPUT_DIR}/{OUTPUT_STEM}.{{npy,parquet}}")
    finally:
        session.close()


if __name__ == "__main__":
    main()
