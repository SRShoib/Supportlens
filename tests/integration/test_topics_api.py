import uuid
from datetime import UTC, datetime, timedelta

import pytest
from api.db import session as session_module
from api.db.models import Prediction, Ticket, TicketSource, Topic
from api.db.session import make_engine
from api.main import app
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

pytestmark = pytest.mark.integration

# 6 consecutive Mondays -- matches tests/unit/test_trend_metrics.py's
# window size, long enough that every week has MIN_HISTORY_WEEKS=4 other
# weeks of history.
WEEK_STARTS = [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(weeks=i) for i in range(6)]
SPIKE_WEEK = WEEK_STARTS[-1].date().isoformat()
MODEL_VERSION = "topics_bertopic_v1"


def _make_tickets_with_topic(
    session: Session, topic_key: int, week_start: datetime, count: int, label: str
) -> None:
    # Two passes, each with its own commit: Prediction has no ORM
    # relationship() back to Ticket (only the raw FK column), so
    # SQLAlchemy's unit-of-work has no dependency edge telling it to flush
    # `tickets` inserts before `predictions` inserts within one flush --
    # committing the tickets first (same pattern every other integration
    # test's _make_ticket helper uses) avoids relying on that ordering.
    tickets = [
        Ticket(
            id=uuid.uuid4(),
            source=TicketSource.TWITTER,
            external_id=str(uuid.uuid4()),
            channel="twitter",
            created_at=week_start,
        )
        for _ in range(count)
    ]
    session.add_all(tickets)
    session.commit()

    session.add_all(
        [
            Prediction(
                ticket_id=ticket.id,
                task="topic",
                label=label,
                score=0.9,
                payload={"topic_key": topic_key, "keywords": []},
                model_version=MODEL_VERSION,
            )
            for ticket in tickets
        ]
    )


def _seed_with_injected_spike(session: Session) -> None:
    session.add(
        Topic(
            topic_key=0,
            label="battery issue",
            keywords=["battery"],
            size=22,
            model_version=MODEL_VERSION,
        )
    )
    session.add(
        Topic(
            topic_key=1,
            label="general chatter",
            keywords=["hello"],
            size=48,
            model_version=MODEL_VERSION,
        )
    )

    # topic 0's real weekly counts spike hard in the last week; topic 1 is a
    # flat background topic supplying most of the volume, so the spike
    # actually moves topic 0's *share*, not just its raw count -- see
    # ml/evaluation/trend_metrics.py's module docstring.
    spiking_counts = [1, 2, 1, 2, 1, 15]
    for week_start, count in zip(WEEK_STARTS, spiking_counts, strict=True):
        _make_tickets_with_topic(session, 0, week_start, count, "battery issue")
    for week_start in WEEK_STARTS:
        _make_tickets_with_topic(session, 1, week_start, 8, "general chatter")
    session.commit()


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, database_url: str) -> TestClient:
    engine = make_engine(database_url)
    monkeypatch.setattr(session_module, "engine", engine)
    monkeypatch.setattr(session_module, "SessionLocal", sessionmaker(bind=engine))
    return TestClient(app)


def test_list_topics_returns_the_full_catalog(db_session: Session, client: TestClient) -> None:
    _seed_with_injected_spike(db_session)

    response = client.get("/topics")

    assert response.status_code == 200
    labels = {t["label"] for t in response.json()}
    assert labels == {"battery issue", "general chatter"}


def test_topic_volume_returns_the_dense_window_and_flags_the_spike(
    db_session: Session, client: TestClient
) -> None:
    _seed_with_injected_spike(db_session)

    response = client.get("/topics/volume")

    assert response.status_code == 200
    body = response.json()
    assert body["weeks"] == [w.date().isoformat() for w in WEEK_STARTS]

    series_by_topic = {s["topic_id"]: s for s in body["series"]}
    assert set(series_by_topic) == {0, 1}

    spike_point = next(p for p in series_by_topic[0]["points"] if p["week"] == SPIKE_WEEK)
    assert spike_point["is_emerging"] is True
    assert spike_point["z_score"] > 2
    assert spike_point["count"] == 15


def test_emerging_endpoint_fires_on_the_injected_spike_and_nothing_else(
    db_session: Session, client: TestClient
) -> None:
    _seed_with_injected_spike(db_session)

    response = client.get("/topics/emerging")

    assert response.status_code == 200
    issues = response.json()
    assert len(issues) == 1
    assert issues[0]["topic_id"] == 0
    assert issues[0]["week"] == SPIKE_WEEK
    assert issues[0]["label"] == "battery issue"


def test_emerging_endpoint_is_empty_when_no_topic_predictions_exist(
    db_session: Session, client: TestClient
) -> None:
    # db_session is requested (but unused directly) purely for its
    # TRUNCATE-before-test side effect (tests/integration/conftest.py) --
    # the container is session-scoped, so without it this test would see
    # whatever an earlier test in this module already committed.
    response = client.get("/topics/emerging")

    assert response.status_code == 200
    assert response.json() == []


def test_topic_volume_excludes_the_outlier_cluster(db_session: Session, client: TestClient) -> None:
    db_session.add(
        Topic(topic_key=-1, label="outliers", keywords=[], size=100, model_version=MODEL_VERSION)
    )
    for week_start in WEEK_STARTS:
        _make_tickets_with_topic(db_session, -1, week_start, 50, "outliers")
    db_session.commit()

    response = client.get("/topics/volume")

    assert response.status_code == 200
    assert response.json() == {"weeks": [], "series": []}
