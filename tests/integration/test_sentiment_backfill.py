import uuid
from pathlib import Path

import pytest
from api.db.models import AuthorRole, Message, Prediction, Ticket, TicketSource
from sqlalchemy import select
from sqlalchemy.orm import Session

from scripts import compute_sentiment_trajectories as backfill

pytestmark = pytest.mark.integration

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "models"
STUB_SENTIMENT_BASELINE = FIXTURES / "stub_sentiment" / "model.joblib"
STUB_URGENCY_TRANSFORMER_DIR = FIXTURES / "stub_transformer_urgency"


@pytest.fixture(autouse=True)
def _use_stub_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backfill, "SENTIMENT_BASELINE_PATH", STUB_SENTIMENT_BASELINE)
    monkeypatch.setattr(backfill, "URGENCY_TRANSFORMER_DIR", STUB_URGENCY_TRANSFORMER_DIR)


def _make_ticket(session: Session, texts_and_roles: list[tuple[str, AuthorRole]]) -> Ticket:
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
    session.add(ticket)
    session.commit()
    return ticket


def test_backfill_writes_one_trajectory_prediction_per_multi_message_ticket(
    db_session: Session,
) -> None:
    ticket = _make_ticket(
        db_session,
        [
            ("this is awful", AuthorRole.CUSTOMER),
            ("please help", AuthorRole.AGENT),
            ("I love it now", AuthorRole.CUSTOMER),
        ],
    )

    written = backfill.compute_and_persist(db_session, [ticket], "baseline")

    assert written == 1
    prediction = db_session.scalars(
        select(Prediction).where(
            Prediction.ticket_id == ticket.id, Prediction.task == "sentiment_trajectory"
        )
    ).one()
    assert len(prediction.payload["sequence"]) == 3
    assert prediction.model_version == "baseline_sentiment_v1"
    assert prediction.payload["urgency_model_version"] == backfill.URGENCY_MODEL_VERSION
    assert "urgency_label" in prediction.payload


def test_backfill_handles_single_message_ticket(db_session: Session) -> None:
    ticket = _make_ticket(db_session, [("this is okay", AuthorRole.CUSTOMER)])

    written = backfill.compute_and_persist(db_session, [ticket], "baseline")

    assert written == 1
    prediction = db_session.scalars(
        select(Prediction).where(Prediction.ticket_id == ticket.id)
    ).one()
    assert len(prediction.payload["sequence"]) == 1


def test_backfill_final_customer_label_ignores_trailing_agent_message(
    db_session: Session,
) -> None:
    ticket = _make_ticket(
        db_session,
        [
            ("this is awful", AuthorRole.CUSTOMER),
            ("I love it now", AuthorRole.AGENT),
        ],
    )

    backfill.compute_and_persist(db_session, [ticket], "baseline")

    prediction = db_session.scalars(
        select(Prediction).where(Prediction.ticket_id == ticket.id)
    ).one()
    assert prediction.label == "negative"


def test_rerunning_backfill_does_not_duplicate_rows(db_session: Session) -> None:
    ticket = _make_ticket(db_session, [("this is awful", AuthorRole.CUSTOMER)])

    backfill.compute_and_persist(db_session, [ticket], "baseline")
    backfill.compute_and_persist(db_session, [ticket], "baseline")

    predictions = db_session.scalars(
        select(Prediction).where(
            Prediction.ticket_id == ticket.id, Prediction.task == "sentiment_trajectory"
        )
    ).all()
    assert len(predictions) == 1


def test_rerunning_backfill_does_not_touch_other_tasks(db_session: Session) -> None:
    ticket = _make_ticket(db_session, [("this is awful", AuthorRole.CUSTOMER)])
    db_session.add(Prediction(ticket_id=ticket.id, task="entities", model_version="v1"))
    db_session.commit()

    backfill.compute_and_persist(db_session, [ticket], "baseline")

    other = db_session.scalars(select(Prediction).where(Prediction.task == "entities")).all()
    assert len(other) == 1


def test_backfill_skips_tickets_with_no_messages(db_session: Session) -> None:
    ticket = Ticket(
        id=uuid.uuid4(), source=TicketSource.TWITTER, external_id="empty", channel="twitter"
    )
    db_session.add(ticket)
    db_session.commit()

    written = backfill.compute_and_persist(db_session, [ticket], "baseline")

    assert written == 0
