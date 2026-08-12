"""Indexes SPEC M8's two Chroma collections.

"resolved_tickets": the retrieval key (embedded text, and Chroma's
`document` field) is a ticket's customer-problem text -- exactly
scripts/compute_embeddings.py's `build_documents`, reused here rather than
reimplemented, restricted first to the resolved subset
(ml/data/resolved_tickets.py). The full thread text -- including the
agent's resolution, which the customer-problem text alone doesn't have --
is carried in `metadata["thread_text"]` so the RAG step
(ml/inference/rag_reply.py) has something to cite from. Search only ever
matches against what's embedded (the problem text); RAG only ever reads the
resolution from metadata.

"kb_articles": embeds `{title}\\n{body}` for every ml/data/kb_generate.py
row in Postgres (the canonical source -- this script only reads it).

Idempotent: chroma ids are the ticket/article Postgres ids directly (both
already stable -- ticket ids are deterministic per ml/data/ids.py, kb
article ids per kb_generate.py's own deterministic scheme), so re-running
after new tickets are ingested or the KB is regenerated is a plain upsert,
never a duplicate.

Run:
  uv run python scripts/index_search_corpus.py
  uv run python scripts/index_search_corpus.py --limit 500   # smoke test
"""

import argparse
import json
import uuid
from typing import Any

from api.db.models import AuthorRole, KbArticle, Ticket
from api.db.session import SessionLocal
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from ml.data.documents import build_documents
from ml.data.resolved_tickets import resolved_ticket_ids
from ml.inference.base import format_dialogue
from ml.inference.vector_store import ChromaVectorStore

RESOLVED_TICKETS_COLLECTION = "resolved_tickets"
KB_ARTICLES_COLLECTION = "kb_articles"
BATCH_SIZE = 64
_SPEAKER_LABEL = {AuthorRole.CUSTOMER: "Customer", AuthorRole.AGENT: "Agent"}


def thread_text(ticket: Ticket) -> str:
    turns = [(_SPEAKER_LABEL[m.author_role], m.text_clean) for m in ticket.messages]
    return format_dialogue(turns)


def build_ticket_metadata(ticket: Ticket) -> dict[str, Any]:
    return {
        "ticket_id": str(ticket.id),
        "thread_text": thread_text(ticket),
        "source": ticket.source.value,
        "created_at": ticket.created_at.isoformat() if ticket.created_at else "",
    }


def build_kb_metadata(article: KbArticle) -> dict[str, Any]:
    return {
        "article_id": str(article.id),
        "title": article.title,
        "tags": json.dumps(article.tags),
        "source_kind": article.source_kind,
        "source_key": article.source_key,
    }


def load_resolved_tickets(session: Session, limit: int | None = None) -> list[Ticket]:
    ids = [uuid.UUID(rid) for rid in resolved_ticket_ids(session)]
    if not ids:
        return []
    stmt = (
        select(Ticket)
        .options(selectinload(Ticket.messages))
        .where(Ticket.id.in_(ids))
        .order_by(Ticket.ingested_at)
    )
    if limit is not None:
        stmt = stmt.limit(limit)
    return list(session.scalars(stmt).all())


def index_resolved_tickets(store: ChromaVectorStore, tickets: list[Ticket]) -> int:
    # Lazy import: sentence-transformers lives behind the `search`
    # dependency group -- same reason scripts/compute_embeddings.py defers
    # this import, so build_ticket_metadata/build_documents above stay
    # importable/unit-testable without it installed.
    from ml.inference.embeddings import SentenceEmbeddingPredictor

    ticket_ids, documents = build_documents(tickets)
    print(f"embedding {len(documents)} resolved tickets ({len(tickets) - len(documents)} skipped)")
    if not documents:
        return 0

    by_id = {str(t.id): t for t in tickets}
    metadatas = [build_ticket_metadata(by_id[tid]) for tid in ticket_ids]

    predictor = SentenceEmbeddingPredictor()
    vectors = predictor.encode(documents, batch_size=BATCH_SIZE)
    store.upsert(
        RESOLVED_TICKETS_COLLECTION,
        ids=ticket_ids,
        embeddings=vectors.tolist(),
        documents=documents,
        metadatas=metadatas,
    )
    return len(ticket_ids)


def index_kb_articles(store: ChromaVectorStore, articles: list[KbArticle]) -> int:
    from ml.inference.embeddings import SentenceEmbeddingPredictor

    print(f"embedding {len(articles)} kb articles")
    if not articles:
        return 0

    ids = [str(a.id) for a in articles]
    documents = [f"{a.title}\n{a.body}" for a in articles]
    metadatas = [build_kb_metadata(a) for a in articles]

    predictor = SentenceEmbeddingPredictor()
    vectors = predictor.encode(documents, batch_size=BATCH_SIZE)
    store.upsert(
        KB_ARTICLES_COLLECTION,
        ids=ids,
        embeddings=vectors.tolist(),
        documents=documents,
        metadatas=metadatas,
    )
    return len(ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap how many resolved tickets to index, oldest-ingested first (default: all)",
    )
    args = parser.parse_args()

    store = ChromaVectorStore()
    session = SessionLocal()
    try:
        tickets = load_resolved_tickets(session, args.limit)
        n_tickets = index_resolved_tickets(store, tickets)
        print(f"wrote {n_tickets} tickets to '{RESOLVED_TICKETS_COLLECTION}'")

        articles = list(session.scalars(select(KbArticle)).all())
        n_articles = index_kb_articles(store, articles)
        print(f"wrote {n_articles} articles to '{KB_ARTICLES_COLLECTION}'")
    finally:
        session.close()


if __name__ == "__main__":
    main()
