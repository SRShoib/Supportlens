import uuid

import pytest
from api.db.models import AuthorRole, Message, Prediction, Ticket, TicketSource
from sqlalchemy.orm import Session

from scripts.index_search_corpus import load_resolved_tickets

pytestmark = pytest.mark.integration


def _ticket(session: Session, source: TicketSource, resolution_quality: float | None) -> Ticket:
    ticket = Ticket(
        id=uuid.uuid4(), source=source, external_id=str(uuid.uuid4()), channel="twitter"
    )
    ticket.messages.append(
        Message(
            id=uuid.uuid4(),
            ticket_id=ticket.id,
            seq=0,
            author_role=AuthorRole.CUSTOMER,
            text_raw="my order is late",
            text_clean="my order is late",
            content_hash=str(uuid.uuid4()),
            external_id="0",
        )
    )
    session.add(ticket)
    session.flush()
    if resolution_quality is not None:
        session.add(
            Prediction(
                ticket_id=ticket.id,
                task="sentiment_trajectory",
                label="positive",
                score=resolution_quality,
                payload={},
                model_version="v1",
            )
        )
    session.commit()
    return ticket


def test_returns_only_resolved_twitter_tickets_with_messages_loaded(db_session: Session) -> None:
    resolved = _ticket(db_session, TicketSource.TWITTER, 0.5)
    _ticket(db_session, TicketSource.TWITTER, -0.5)
    _ticket(db_session, TicketSource.BITEXT, 0.9)

    tickets = load_resolved_tickets(db_session)

    assert [t.id for t in tickets] == [resolved.id]
    assert tickets[0].messages[0].text_clean == "my order is late"


def test_returns_empty_list_when_nothing_is_resolved(db_session: Session) -> None:
    _ticket(db_session, TicketSource.TWITTER, -0.1)

    assert load_resolved_tickets(db_session) == []


def test_limit_caps_the_number_of_tickets_returned(db_session: Session) -> None:
    for _ in range(3):
        _ticket(db_session, TicketSource.TWITTER, 0.5)

    tickets = load_resolved_tickets(db_session, limit=2)

    assert len(tickets) == 2
