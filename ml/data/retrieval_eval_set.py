"""SPEC M8's "100-query synthetic eval set (question -> known relevant
ticket)" for retrieval hit-rate@5. Zero-cost, deterministic, no LLM: samples
100 resolved tickets (RANDOM_SEED, ml.data.resolved_tickets) that have >= 2
customer messages, and uses each ticket's FIRST customer message as the
query. The ticket's own indexed document
(scripts/index_search_corpus.py::index_resolved_tickets) is the
concatenation of ALL its customer messages, so for every sampled ticket the
query is a genuine partial view of the indexed text, never identical to
it -- tickets with exactly 1 customer message are excluded specifically to
avoid that trivial identical-text case, see docs/decisions.md.

Ground truth: the ticket's own id is the "known relevant ticket" SPEC M8
asks for -- not a held-out human relevance judgment (no budget/scope in
SPEC M8 for that). A documented limitation, same shortcut class as every
other weak-supervision step in this project (SPEC §7).

Run: uv run python -m ml.data.retrieval_eval_set
"""

import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from api.db.models import AuthorRole, Ticket
from api.db.session import SessionLocal

from scripts.index_search_corpus import load_resolved_tickets

OUTPUT_PATH = Path("data/eval/retrieval_queries.parquet")
SAMPLE_SIZE = 100
RANDOM_SEED = 42
MIN_CUSTOMER_MESSAGES = 2


@dataclass(frozen=True)
class RetrievalQuery:
    query: str
    relevant_ticket_id: str


def _customer_message_count(ticket: Ticket) -> int:
    return sum(1 for m in ticket.messages if m.author_role == AuthorRole.CUSTOMER)


def eligible_tickets(tickets: list[Ticket]) -> list[Ticket]:
    return [t for t in tickets if _customer_message_count(t) >= MIN_CUSTOMER_MESSAGES]


def first_customer_message(ticket: Ticket) -> str:
    for message in ticket.messages:  # relationship is order_by=Message.seq
        if message.author_role == AuthorRole.CUSTOMER:
            return message.text_clean
    raise ValueError(f"ticket {ticket.id} has no customer message")


def build_eval_set(
    tickets: list[Ticket], sample_size: int = SAMPLE_SIZE, seed: int = RANDOM_SEED
) -> list[RetrievalQuery]:
    eligible = eligible_tickets(tickets)
    sampled = random.Random(seed).sample(eligible, min(sample_size, len(eligible)))
    return [
        RetrievalQuery(query=first_customer_message(t), relevant_ticket_id=str(t.id))
        for t in sampled
    ]


def main() -> None:
    session = SessionLocal()
    try:
        tickets = load_resolved_tickets(session, limit=None)
        eval_set = build_eval_set(tickets)
    finally:
        session.close()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"query": q.query, "relevant_ticket_id": q.relevant_ticket_id} for q in eval_set]
    ).to_parquet(OUTPUT_PATH)
    print(f"wrote {len(eval_set)} queries to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
