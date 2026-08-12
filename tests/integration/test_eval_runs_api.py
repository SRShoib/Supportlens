from datetime import UTC, datetime, timedelta

import pytest
from api.db import session as session_module
from api.db.models import EvalRun
from api.db.session import make_engine
from api.main import app
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

pytestmark = pytest.mark.integration

_BASE_TIME = datetime(2024, 1, 1, tzinfo=UTC)


def _add_run(
    session: Session,
    *,
    task: str,
    model_version: str,
    dataset: str,
    split: str,
    started_at: datetime = _BASE_TIME,
) -> None:
    session.add(
        EvalRun(
            task=task,
            model_version=model_version,
            dataset=dataset,
            split=split,
            metrics={"macro_f1": 0.9},
            params={},
            status="completed",
            started_at=started_at,
        )
    )
    session.commit()


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, database_url: str) -> TestClient:
    engine = make_engine(database_url)
    monkeypatch.setattr(session_module, "engine", engine)
    monkeypatch.setattr(session_module, "SessionLocal", sessionmaker(bind=engine))
    return TestClient(app)


def test_list_eval_runs_returns_every_run_newest_first(
    db_session: Session, client: TestClient
) -> None:
    _add_run(
        db_session,
        task="intent",
        model_version="baseline_intent_v1",
        dataset="bitext",
        split="test",
        started_at=_BASE_TIME,
    )
    _add_run(
        db_session,
        task="urgency",
        model_version="baseline_urgency_v1",
        dataset="twitter_slice_v1",
        split="test",
        started_at=_BASE_TIME + timedelta(hours=1),
    )

    response = client.get("/eval-runs")

    assert response.status_code == 200
    body = response.json()
    assert {r["task"] for r in body} == {"intent", "urgency"}
    assert body[0]["task"] == "urgency"


def test_list_eval_runs_filters_by_task(db_session: Session, client: TestClient) -> None:
    _add_run(
        db_session,
        task="intent",
        model_version="baseline_intent_v1",
        dataset="bitext",
        split="test",
    )
    _add_run(
        db_session,
        task="urgency",
        model_version="baseline_urgency_v1",
        dataset="twitter_slice_v1",
        split="test",
    )

    response = client.get("/eval-runs", params={"task": "intent"})

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["task"] == "intent"


def test_list_eval_runs_filters_by_model_version(db_session: Session, client: TestClient) -> None:
    _add_run(
        db_session,
        task="intent",
        model_version="baseline_intent_v1",
        dataset="bitext",
        split="test",
    )
    _add_run(
        db_session,
        task="intent",
        model_version="transformer_distilbert-base-uncased_v1",
        dataset="bitext",
        split="test",
    )

    response = client.get("/eval-runs", params={"model_version": "baseline_intent_v1"})

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["model_version"] == "baseline_intent_v1"


def test_list_eval_runs_respects_limit(db_session: Session, client: TestClient) -> None:
    for i in range(5):
        _add_run(db_session, task="intent", model_version=f"v{i}", dataset="bitext", split="test")

    response = client.get("/eval-runs", params={"limit": 2})

    assert response.status_code == 200
    assert len(response.json()) == 2


def test_list_eval_runs_is_empty_when_nothing_persisted(
    db_session: Session, client: TestClient
) -> None:
    response = client.get("/eval-runs")

    assert response.status_code == 200
    assert response.json() == []
