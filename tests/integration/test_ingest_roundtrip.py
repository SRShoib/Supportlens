import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from api.db import session as session_module
from api.db.models import Message, Prediction, Ticket
from api.db.session import make_engine
from api.main import app
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from ml.data.loaders import bitext, twitter
from ml.data.persist import persist_tickets

pytestmark = pytest.mark.integration

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
TWCS_FIXTURE = FIXTURES / "twcs_sample.csv"


def _bitext_rows() -> list[bitext.BitextRow]:
    with (FIXTURES / "bitext_sample.jsonl").open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


@pytest.fixture
def api_client(migrated_db: str, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    engine = make_engine(migrated_db)
    monkeypatch.setattr(session_module, "engine", engine)
    monkeypatch.setattr(session_module, "SessionLocal", sessionmaker(bind=engine))
    with TestClient(app) as client:
        yield client


def test_fixtures_ingest_with_correct_row_counts(db_session: Session) -> None:
    n_bitext = persist_tickets(db_session, bitext.iter_tickets(rows=_bitext_rows()))
    n_twitter = persist_tickets(db_session, twitter.iter_tickets(TWCS_FIXTURE))

    assert n_bitext == 5
    assert n_twitter == 7

    assert db_session.query(Ticket).count() == n_bitext + n_twitter
    assert db_session.query(Message).count() == 5 * 2 + 12  # bitext: 2/ticket; twcs: 12 rows total


def test_second_ingest_is_a_noop(db_session: Session) -> None:
    persist_tickets(db_session, bitext.iter_tickets(rows=_bitext_rows()))
    first_tickets = db_session.query(Ticket).count()
    first_messages = db_session.query(Message).count()

    persist_tickets(db_session, bitext.iter_tickets(rows=_bitext_rows()))

    assert db_session.query(Ticket).count() == first_tickets
    assert db_session.query(Message).count() == first_messages


def test_second_ingest_of_twitter_fixture_is_a_noop(db_session: Session) -> None:
    persist_tickets(db_session, twitter.iter_tickets(TWCS_FIXTURE))
    first_tickets = db_session.query(Ticket).count()

    persist_tickets(db_session, twitter.iter_tickets(TWCS_FIXTURE))

    assert db_session.query(Ticket).count() == first_tickets


def test_cascade_delete_removes_messages(db_session: Session) -> None:
    persist_tickets(db_session, bitext.iter_tickets(rows=_bitext_rows()[:1]))
    ticket = db_session.query(Ticket).one()
    assert db_session.query(Message).filter_by(ticket_id=ticket.id).count() == 2

    db_session.delete(ticket)
    db_session.commit()

    assert db_session.query(Message).filter_by(ticket_id=ticket.id).count() == 0


def test_messages_returned_in_seq_order(db_session: Session) -> None:
    persist_tickets(db_session, twitter.iter_tickets(TWCS_FIXTURE))

    ticket = db_session.query(Ticket).filter_by(external_id="102").one()
    seqs = [m.seq for m in ticket.messages]

    assert seqs == [0, 1, 2]


def test_api_lists_tickets_filtered_by_source_with_pagination(
    db_session: Session, api_client: TestClient
) -> None:
    persist_tickets(db_session, bitext.iter_tickets(rows=_bitext_rows()))
    persist_tickets(db_session, twitter.iter_tickets(TWCS_FIXTURE))

    page1 = api_client.get("/tickets", params={"source": "bitext", "limit": 3})
    assert page1.status_code == 200
    page1_body = page1.json()
    assert len(page1_body) == 3
    assert all(t["source"] == "bitext" for t in page1_body)

    page2 = api_client.get("/tickets", params={"source": "bitext", "limit": 3, "offset": 3})
    page2_body = page2.json()
    assert len(page2_body) == 2

    page1_ids = {t["id"] for t in page1_body}
    page2_ids = {t["id"] for t in page2_body}
    assert page1_ids.isdisjoint(page2_ids)


def test_api_get_single_ticket_returns_messages(
    db_session: Session, api_client: TestClient
) -> None:
    persist_tickets(db_session, bitext.iter_tickets(rows=_bitext_rows()[:1]))
    ticket = db_session.query(Ticket).one()

    response = api_client.get(f"/tickets/{ticket.id}")

    assert response.status_code == 200
    body = response.json()
    assert len(body["messages"]) == 2


def test_api_get_unknown_ticket_returns_404(api_client: TestClient) -> None:
    response = api_client.get("/tickets/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404


def test_api_lists_ticket_predictions_filtered_by_task(
    db_session: Session, api_client: TestClient
) -> None:
    persist_tickets(db_session, bitext.iter_tickets(rows=_bitext_rows()[:1]))
    ticket = db_session.query(Ticket).one()
    db_session.add_all(
        [
            Prediction(
                ticket_id=ticket.id,
                task="sentiment_trajectory",
                label="positive",
                score=0.5,
                payload={"sequence": ["negative", "positive"]},
                model_version="transformer_sentiment_distilbert-base-uncased_v1",
            ),
            Prediction(
                ticket_id=ticket.id,
                task="some_other_task",
                model_version="v1",
            ),
        ]
    )
    db_session.commit()

    response = api_client.get(
        f"/tickets/{ticket.id}/predictions", params={"task": "sentiment_trajectory"}
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["task"] == "sentiment_trajectory"
    assert body[0]["payload"]["sequence"] == ["negative", "positive"]


def test_api_lists_all_ticket_predictions_when_task_not_specified(
    db_session: Session, api_client: TestClient
) -> None:
    persist_tickets(db_session, bitext.iter_tickets(rows=_bitext_rows()[:1]))
    ticket = db_session.query(Ticket).one()
    db_session.add_all(
        [
            Prediction(ticket_id=ticket.id, task="sentiment_trajectory", model_version="v1"),
            Prediction(ticket_id=ticket.id, task="some_other_task", model_version="v1"),
        ]
    )
    db_session.commit()

    response = api_client.get(f"/tickets/{ticket.id}/predictions")

    assert response.status_code == 200
    assert len(response.json()) == 2


def test_api_ticket_predictions_empty_list_when_none_persisted_yet(
    db_session: Session, api_client: TestClient
) -> None:
    persist_tickets(db_session, bitext.iter_tickets(rows=_bitext_rows()[:1]))
    ticket = db_session.query(Ticket).one()

    response = api_client.get(f"/tickets/{ticket.id}/predictions")

    assert response.status_code == 200
    assert response.json() == []


def test_api_ticket_predictions_404_for_unknown_ticket(api_client: TestClient) -> None:
    response = api_client.get("/tickets/00000000-0000-0000-0000-000000000000/predictions")
    assert response.status_code == 404
