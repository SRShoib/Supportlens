"""The customer-messages-only embedding unit (SPEC M7/M8: embed a ticket's
customer text, never the full agent+customer thread -- see
docs/decisions.md). Split out of scripts/compute_embeddings.py (M10): that
module imports pandas/numpy at the top level for its own bulk-write path,
but build_documents() itself is a pure function with no heavy dependency --
apps/api/routers/rag.py needs it at request time (to build the query text
for a suggested-reply draft) and previously imported it straight from
scripts.compute_embeddings, silently dragging pandas into the API's import
graph and crashing the API container outright the first time it actually
ran under `docker compose up` (pandas lives in the `ml` dependency group,
excluded from the API image). Living here instead keeps every caller
-- scripts/compute_embeddings.py, scripts/index_search_corpus.py,
apps/api/routers/rag.py -- importable without pandas/numpy.
"""

from api.db.models import AuthorRole, Ticket


def _customer_document(ticket: Ticket) -> str | None:
    customer_texts = [m.text_clean for m in ticket.messages if m.author_role == AuthorRole.CUSTOMER]
    if not customer_texts:
        return None
    return "\n".join(customer_texts)


def build_documents(tickets: list[Ticket]) -> tuple[list[str], list[str]]:
    """Returns (ticket_ids, documents), skipping tickets with no customer
    message -- the two lists stay index-aligned with each other and with
    whatever embeddings array the caller computes from `documents`."""
    ticket_ids: list[str] = []
    documents: list[str] = []
    for ticket in tickets:
        document = _customer_document(ticket)
        if document is None:
            continue
        ticket_ids.append(str(ticket.id))
        documents.append(document)
    return ticket_ids, documents
