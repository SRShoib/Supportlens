import uuid

from api.db.models import AuthorRole, KbArticle, Message, Ticket, TicketSource

from scripts.index_search_corpus import build_kb_metadata, build_ticket_metadata, thread_text


def _ticket(texts_and_roles: list[tuple[str, AuthorRole]], created_at=None) -> Ticket:
    ticket = Ticket(
        id=uuid.uuid4(),
        source=TicketSource.TWITTER,
        external_id=str(uuid.uuid4()),
        channel="twitter",
        created_at=created_at,
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


def test_thread_text_labels_speakers_and_keeps_chronological_order() -> None:
    ticket = _ticket(
        [
            ("my order is late", AuthorRole.CUSTOMER),
            ("sorry, can you DM us your order number", AuthorRole.AGENT),
            ("here it is: 12345", AuthorRole.CUSTOMER),
        ]
    )

    assert thread_text(ticket) == (
        "Customer: my order is late\n"
        "Agent: sorry, can you DM us your order number\n"
        "Customer: here it is: 12345"
    )


def test_build_ticket_metadata_includes_thread_text_and_source() -> None:
    ticket = _ticket([("battery drains fast", AuthorRole.CUSTOMER)])

    metadata = build_ticket_metadata(ticket)

    assert metadata["ticket_id"] == str(ticket.id)
    assert metadata["thread_text"] == "Customer: battery drains fast"
    assert metadata["source"] == "twitter"
    assert metadata["created_at"] == ""


def test_build_kb_metadata_serializes_tags_as_json() -> None:
    article = KbArticle(
        id=uuid.uuid4(),
        title="How to Cancel an Order",
        body="Some body text.",
        tags=["cancel_order", "orders"],
        source_kind="intent",
        source_key="cancel_order",
        generator_version="kb_template_v1",
    )

    metadata = build_kb_metadata(article)

    assert metadata["article_id"] == str(article.id)
    assert metadata["title"] == "How to Cancel an Order"
    assert metadata["tags"] == '["cancel_order", "orders"]'
    assert metadata["source_kind"] == "intent"
    assert metadata["source_key"] == "cancel_order"
