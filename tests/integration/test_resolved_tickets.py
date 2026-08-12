import uuid

import pytest
from api.db.models import Prediction, Ticket, TicketSource
from sqlalchemy.orm import Session

from ml.data.resolved_tickets import resolved_ticket_ids

pytestmark = pytest.mark.integration


def _ticket(session: Session, source: TicketSource) -> Ticket:
    ticket = Ticket(
        id=uuid.uuid4(), source=source, external_id=str(uuid.uuid4()), channel="twitter"
    )
    session.add(ticket)
    session.commit()
    return ticket


def _trajectory_prediction(
    ticket: Ticket, score: float, task: str = "sentiment_trajectory"
) -> Prediction:
    return Prediction(
        ticket_id=ticket.id,
        task=task,
        label="positive",
        score=score,
        payload={},
        model_version="v1",
    )


def test_includes_twitter_ticket_with_positive_resolution_quality(db_session: Session) -> None:
    ticket = _ticket(db_session, TicketSource.TWITTER)
    db_session.add(_trajectory_prediction(ticket, 0.5))
    db_session.commit()

    assert resolved_ticket_ids(db_session) == [str(ticket.id)]


def test_excludes_twitter_ticket_with_zero_or_negative_resolution_quality(
    db_session: Session,
) -> None:
    zero = _ticket(db_session, TicketSource.TWITTER)
    negative = _ticket(db_session, TicketSource.TWITTER)
    db_session.add(_trajectory_prediction(zero, 0.0))
    db_session.add(_trajectory_prediction(negative, -0.3))
    db_session.commit()

    assert resolved_ticket_ids(db_session) == []


def test_excludes_bitext_ticket_even_with_positive_resolution_quality(db_session: Session) -> None:
    ticket = _ticket(db_session, TicketSource.BITEXT)
    db_session.add(_trajectory_prediction(ticket, 0.9))
    db_session.commit()

    assert resolved_ticket_ids(db_session) == []


def test_excludes_twitter_ticket_with_no_sentiment_trajectory_prediction(
    db_session: Session,
) -> None:
    _ticket(db_session, TicketSource.TWITTER)

    assert resolved_ticket_ids(db_session) == []


def test_excludes_prediction_from_a_different_task(db_session: Session) -> None:
    ticket = _ticket(db_session, TicketSource.TWITTER)
    db_session.add(_trajectory_prediction(ticket, 0.9, task="topic"))
    db_session.commit()

    assert resolved_ticket_ids(db_session) == []
