import uuid

from api.db.models import AuthorRole, Message, Ticket, TicketSource

from ml.data.documents import build_documents


def _ticket(texts_and_roles: list[tuple[str, AuthorRole]]) -> Ticket:
    ticket = Ticket(
        id=uuid.uuid4(),
        source=TicketSource.TWITTER,
        external_id=str(uuid.uuid4()),
        channel="twitter",
    )
    for seq, (text, role) in enumerate(texts_and_roles):
        ticket.messages.append(
            Message(
                id=uuid.uuid4(),
                ticket_id=ticket.id,
                seq=seq,
                author_role=role,
                text_raw=text,
                text_clean=text,
                content_hash=str(uuid.uuid4()),
                external_id=str(seq),
            )
        )
    return ticket


def test_build_documents_joins_customer_messages_only_in_order() -> None:
    ticket = _ticket(
        [
            ("my order is late", AuthorRole.CUSTOMER),
            ("sorry to hear that, can you DM us", AuthorRole.AGENT),
            ("it still hasn't arrived", AuthorRole.CUSTOMER),
        ]
    )

    ticket_ids, documents = build_documents([ticket])

    assert ticket_ids == [str(ticket.id)]
    assert documents == ["my order is late\nit still hasn't arrived"]


def test_build_documents_skips_tickets_with_no_customer_message() -> None:
    ticket = _ticket([("welcome, how can we help", AuthorRole.AGENT)])

    ticket_ids, documents = build_documents([ticket])

    assert ticket_ids == []
    assert documents == []


def test_build_documents_stays_index_aligned_across_multiple_tickets() -> None:
    kept = _ticket([("battery drains fast", AuthorRole.CUSTOMER)])
    skipped = _ticket([("welcome", AuthorRole.AGENT)])
    also_kept = _ticket([("refund please", AuthorRole.CUSTOMER)])

    ticket_ids, documents = build_documents([kept, skipped, also_kept])

    assert ticket_ids == [str(kept.id), str(also_kept.id)]
    assert documents == ["battery drains fast", "refund please"]


def test_build_documents_empty_input_returns_empty_lists() -> None:
    assert build_documents([]) == ([], [])
