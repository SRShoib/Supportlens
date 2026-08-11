import uuid

import pytest
from api.db.models import AuthorRole, Message, Prediction, Ticket, TicketSource
from sqlalchemy import select
from sqlalchemy.orm import Session

pytestmark = pytest.mark.integration


def _make_message(session: Session) -> Message:
    ticket = Ticket(
        id=uuid.uuid4(),
        source=TicketSource.TWITTER,
        external_id="1",
        channel="twitter",
    )
    message = Message(
        id=uuid.uuid4(),
        ticket_id=ticket.id,
        seq=0,
        author_role=AuthorRole.CUSTOMER,
        text_raw="order ORD-99321 shipped yesterday",
        text_clean="order ORD-99321 shipped yesterday",
        content_hash="deadbeef",
        external_id="1",
    )
    ticket.messages.append(message)
    session.add(ticket)
    session.commit()
    return message


def test_entity_spans_fit_in_prediction_payload_with_no_migration(db_session: Session) -> None:
    # Evidence for the M4 design claim that entity spans need no Alembic
    # migration: Prediction already has nullable label/score, a free-text
    # task column, and a JSONB payload -- confirmed here against real
    # Postgres rather than merely asserted.
    message = _make_message(db_session)

    payload = {
        "entities": [
            {"start": 6, "end": 15, "label": "ORDER_ID", "text": "ORD-99321", "score": 1.0},
            {"start": 24, "end": 33, "label": "DATE", "text": "yesterday", "score": 1.0},
        ],
        "truncated": False,
    }
    prediction = Prediction(
        message_id=message.id,
        task="entities",
        label=None,
        score=None,
        payload=payload,
        model_version="rules_ner_v1",
    )
    db_session.add(prediction)
    db_session.commit()

    stored = db_session.scalars(select(Prediction).where(Prediction.task == "entities")).one()
    assert stored.message_id == message.id
    assert stored.ticket_id is None
    assert stored.label is None
    assert stored.score is None
    assert stored.model_version == "rules_ner_v1"
    assert stored.payload == payload
    assert stored.payload["entities"][0]["label"] == "ORDER_ID"
    assert stored.payload["entities"][1]["text"] == "yesterday"


def test_entity_prediction_satisfies_the_has_target_constraint_via_message_id_alone(
    db_session: Session,
) -> None:
    message = _make_message(db_session)

    # ticket_id intentionally omitted entirely -- message_id alone must
    # satisfy the prediction_has_target CheckConstraint.
    prediction = Prediction(
        message_id=message.id,
        task="entities",
        payload={"entities": []},
        model_version="rules_ner_v1",
    )
    db_session.add(prediction)
    db_session.commit()  # would raise IntegrityError if the constraint failed

    stored = db_session.scalars(select(Prediction).where(Prediction.id == prediction.id)).one()
    assert stored.ticket_id is None
    assert stored.message_id == message.id
