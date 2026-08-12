import uuid

import pytest
from api.db.models import Prediction, Ticket, TicketSource, Topic
from sqlalchemy import select
from sqlalchemy.orm import Session

from scripts.assign_topics import TopicAssignment, TopicCatalogEntry, persist_topic_assignments

pytestmark = pytest.mark.integration


def _make_ticket(session: Session) -> Ticket:
    ticket = Ticket(
        id=uuid.uuid4(),
        source=TicketSource.TWITTER,
        external_id=str(uuid.uuid4()),
        channel="twitter",
    )
    session.add(ticket)
    session.commit()
    return ticket


CATALOG = [
    TopicCatalogEntry(topic_key=-1, label="outliers", keywords=[], size=1),
    TopicCatalogEntry(topic_key=0, label="refund, order", keywords=["refund", "order"], size=1),
]


def test_persist_topic_assignments_writes_catalog_and_predictions(db_session: Session) -> None:
    ticket = _make_ticket(db_session)
    assignments = [TopicAssignment(ticket_id=str(ticket.id), topic_key=0, probability=0.87)]

    written = persist_topic_assignments(db_session, "topics_bertopic_v1", CATALOG, assignments)

    assert written == 1
    topics = db_session.scalars(select(Topic)).all()
    assert {t.topic_key for t in topics} == {-1, 0}
    assert all(t.model_version == "topics_bertopic_v1" for t in topics)

    prediction = db_session.scalars(
        select(Prediction).where(Prediction.ticket_id == ticket.id, Prediction.task == "topic")
    ).one()
    assert prediction.label == "refund, order"
    assert prediction.score == pytest.approx(0.87)
    assert prediction.payload == {"topic_key": 0, "keywords": ["refund", "order"]}
    assert prediction.model_version == "topics_bertopic_v1"


def test_rerunning_does_not_duplicate_rows(db_session: Session) -> None:
    ticket = _make_ticket(db_session)
    assignments = [TopicAssignment(ticket_id=str(ticket.id), topic_key=0, probability=0.5)]

    persist_topic_assignments(db_session, "topics_bertopic_v1", CATALOG, assignments)
    persist_topic_assignments(db_session, "topics_bertopic_v1", CATALOG, assignments)

    predictions = db_session.scalars(select(Prediction).where(Prediction.task == "topic")).all()
    topics = db_session.scalars(select(Topic)).all()
    assert len(predictions) == 1
    assert len(topics) == len(CATALOG)


def test_rerunning_does_not_touch_other_tasks(db_session: Session) -> None:
    ticket = _make_ticket(db_session)
    db_session.add(Prediction(ticket_id=ticket.id, task="sentiment_trajectory", model_version="v1"))
    db_session.commit()

    persist_topic_assignments(
        db_session,
        "topics_bertopic_v1",
        CATALOG,
        [TopicAssignment(ticket_id=str(ticket.id), topic_key=0, probability=0.5)],
    )

    other = db_session.scalars(
        select(Prediction).where(Prediction.task == "sentiment_trajectory")
    ).all()
    assert len(other) == 1


def test_switching_variants_fully_replaces_the_previous_catalog(db_session: Session) -> None:
    ticket = _make_ticket(db_session)
    persist_topic_assignments(
        db_session,
        "topics_kmeans_v1",
        CATALOG,
        [TopicAssignment(ticket_id=str(ticket.id), topic_key=0, probability=0.5)],
    )

    new_catalog = [
        TopicCatalogEntry(topic_key=1, label="battery, charge", keywords=["battery"], size=1)
    ]
    persist_topic_assignments(
        db_session,
        "topics_bertopic_v1",
        new_catalog,
        [TopicAssignment(ticket_id=str(ticket.id), topic_key=1, probability=0.9)],
    )

    topics = db_session.scalars(select(Topic)).all()
    assert {(t.model_version, t.topic_key) for t in topics} == {("topics_bertopic_v1", 1)}


def test_assignment_for_unknown_topic_key_falls_back_to_unlabeled(db_session: Session) -> None:
    ticket = _make_ticket(db_session)
    assignments = [TopicAssignment(ticket_id=str(ticket.id), topic_key=999, probability=0.3)]

    persist_topic_assignments(db_session, "topics_bertopic_v1", CATALOG, assignments)

    prediction = db_session.scalars(
        select(Prediction).where(Prediction.ticket_id == ticket.id)
    ).one()
    assert prediction.label == "unlabeled"
    assert prediction.payload == {"topic_key": 999, "keywords": []}
