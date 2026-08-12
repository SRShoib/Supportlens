import uuid
from pathlib import Path

import pytest
from api.db.models import AuthorRole, Message, Prediction, Ticket, TicketSource
from sqlalchemy import select
from sqlalchemy.orm import Session

from scripts import compute_thread_summaries as backfill

pytestmark = pytest.mark.integration

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "models"
STUB_SUMMARY_TRANSFORMER_DIR = FIXTURES / "stub_transformer_thread_summary"


@pytest.fixture(autouse=True)
def _use_stub_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backfill, "SUMMARY_TRANSFORMER_DIR", STUB_SUMMARY_TRANSFORMER_DIR)


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


def test_backfill_writes_one_summary_prediction_per_multi_message_ticket(
    db_session: Session,
) -> None:
    ticket = _make_ticket(
        db_session,
        [
            ("my order is late", AuthorRole.CUSTOMER),
            ("sorry about that", AuthorRole.AGENT),
            ("when will it arrive", AuthorRole.CUSTOMER),
        ],
    )

    written = backfill.compute_and_persist(db_session, [ticket], "baseline")

    assert written == 1
    prediction = db_session.scalars(
        select(Prediction).where(
            Prediction.ticket_id == ticket.id, Prediction.task == "thread_summary"
        )
    ).one()
    assert prediction.model_version == backfill.BASELINE_MODEL_VERSION
    assert prediction.label is not None
    assert prediction.label != ""
    assert prediction.payload["message_count"] == 3


def test_backfill_skips_single_message_tickets(db_session: Session) -> None:
    ticket = _make_ticket(db_session, [("just one message", AuthorRole.CUSTOMER)])

    written = backfill.compute_and_persist(db_session, [ticket], "baseline")

    assert written == 0
    predictions = db_session.scalars(
        select(Prediction).where(Prediction.ticket_id == ticket.id)
    ).all()
    assert predictions == []


def test_backfill_skips_tickets_with_no_messages(db_session: Session) -> None:
    ticket = Ticket(
        id=uuid.uuid4(), source=TicketSource.TWITTER, external_id="empty", channel="twitter"
    )
    db_session.add(ticket)
    db_session.commit()

    written = backfill.compute_and_persist(db_session, [ticket], "baseline")

    assert written == 0


def test_backfill_transformer_model_uses_the_stub_export(db_session: Session) -> None:
    ticket = _make_ticket(
        db_session,
        [("order shipped yesterday", AuthorRole.CUSTOMER), ("please help", AuthorRole.AGENT)],
    )

    backfill.compute_and_persist(db_session, [ticket], "transformer")

    prediction = db_session.scalars(
        select(Prediction).where(Prediction.ticket_id == ticket.id)
    ).one()
    assert prediction.model_version == backfill.TRANSFORMER_MODEL_VERSION


def test_rerunning_backfill_does_not_duplicate_rows(db_session: Session) -> None:
    ticket = _make_ticket(
        db_session, [("hello", AuthorRole.CUSTOMER), ("hi there", AuthorRole.AGENT)]
    )

    backfill.compute_and_persist(db_session, [ticket], "baseline")
    backfill.compute_and_persist(db_session, [ticket], "baseline")

    predictions = db_session.scalars(
        select(Prediction).where(
            Prediction.ticket_id == ticket.id, Prediction.task == "thread_summary"
        )
    ).all()
    assert len(predictions) == 1


def test_reprocessing_a_subset_leaves_other_tickets_summaries_intact(db_session: Session) -> None:
    # A --limit run reprocesses only some tickets -- it must not wipe out
    # every other ticket's already-computed thread_summary the way a
    # global "delete every task=thread_summary row" would.
    ticket_a = _make_ticket(
        db_session, [("order is late", AuthorRole.CUSTOMER), ("sorry", AuthorRole.AGENT)]
    )
    ticket_b = _make_ticket(
        db_session, [("refund please", AuthorRole.CUSTOMER), ("done", AuthorRole.AGENT)]
    )
    backfill.compute_and_persist(db_session, [ticket_a, ticket_b], "baseline")
    original_b = db_session.scalars(
        select(Prediction).where(Prediction.ticket_id == ticket_b.id)
    ).one()

    backfill.compute_and_persist(db_session, [ticket_a], "transformer")

    a_prediction = db_session.scalars(
        select(Prediction).where(Prediction.ticket_id == ticket_a.id)
    ).one()
    b_prediction = db_session.scalars(
        select(Prediction).where(Prediction.ticket_id == ticket_b.id)
    ).one()
    assert a_prediction.model_version == backfill.TRANSFORMER_MODEL_VERSION
    assert b_prediction.model_version == backfill.BASELINE_MODEL_VERSION
    assert b_prediction.id == original_b.id  # untouched, not deleted+reinserted


def test_rerunning_backfill_does_not_touch_other_tasks(db_session: Session) -> None:
    ticket = _make_ticket(
        db_session, [("hello", AuthorRole.CUSTOMER), ("hi there", AuthorRole.AGENT)]
    )
    db_session.add(Prediction(ticket_id=ticket.id, task="sentiment_trajectory", model_version="v1"))
    db_session.commit()

    backfill.compute_and_persist(db_session, [ticket], "baseline")

    other = db_session.scalars(
        select(Prediction).where(Prediction.task == "sentiment_trajectory")
    ).all()
    assert len(other) == 1
