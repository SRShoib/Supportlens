from collections.abc import Iterable, Iterator, Sequence
from itertools import islice
from typing import Any

from api.db.models import Message, Ticket
from api.schemas.ticket import CanonicalTicket
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

# Postgres caps bind parameters at 65535 per statement. Chunking rows here
# (independent of how many tickets are pulled per batch_size) keeps every
# INSERT well under that regardless of column count.
_MAX_ROWS_PER_INSERT = 1000


def _batched(tickets: Iterable[CanonicalTicket], size: int) -> Iterator[list[CanonicalTicket]]:
    it = iter(tickets)
    while batch := list(islice(it, size)):
        yield batch


def _chunked(rows: Sequence[dict[str, Any]], size: int) -> Iterator[Sequence[dict[str, Any]]]:
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


def persist_tickets(
    session: Session, tickets: Iterable[CanonicalTicket], *, batch_size: int = 5000
) -> int:
    """Bulk-insert canonical tickets + messages. Idempotent: ON CONFLICT DO NOTHING
    on the same unique constraints the schema already enforces, so re-running a
    loader against data already in the DB is a no-op rather than a duplicate-row
    generator or an error."""
    total = 0
    for batch in _batched(tickets, batch_size):
        ticket_rows = [
            {
                "id": t.id,
                "source": t.source,
                "external_id": t.external_id,
                "created_at": t.created_at,
                "channel": t.channel,
                "customer_id": t.customer_id,
                "brand": t.brand,
                "lang": t.lang,
                "meta": t.meta,
            }
            for t in batch
        ]
        for chunk in _chunked(ticket_rows, _MAX_ROWS_PER_INSERT):
            stmt = (
                pg_insert(Ticket)
                .values(list(chunk))
                .on_conflict_do_nothing(index_elements=["source", "external_id"])
            )
            session.execute(stmt)

        message_rows = [
            {
                "id": m.id,
                "ticket_id": t.id,
                "seq": m.seq,
                "author_role": m.author_role,
                "text_raw": m.text_raw,
                "text_clean": m.text_clean,
                "sent_at": m.sent_at,
                "lang": m.lang,
                "lang_confidence": m.lang_confidence,
                "content_hash": m.content_hash,
                "external_id": m.external_id,
                "meta": m.meta,
            }
            for t in batch
            for m in t.messages
        ]
        for chunk in _chunked(message_rows, _MAX_ROWS_PER_INSERT):
            stmt = (
                pg_insert(Message)
                .values(list(chunk))
                .on_conflict_do_nothing(index_elements=["ticket_id", "seq"])
            )
            session.execute(stmt)

        session.commit()
        total += len(batch)
    return total
