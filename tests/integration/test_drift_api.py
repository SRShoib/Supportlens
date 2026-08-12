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


def _add_drift_run(
    session: Session,
    *,
    task: str,
    split: str,
    metrics: dict,
    started_at: datetime,
    model_version: str = "all-MiniLM-L6-v2",
) -> None:
    session.add(
        EvalRun(
            task=task,
            model_version=model_version,
            dataset="twitter_slice_v1",
            split=split,
            metrics=metrics,
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


def test_get_drift_returns_all_four_latest_runs(db_session: Session, client: TestClient) -> None:
    _add_drift_run(
        db_session,
        task="drift_embedding",
        split="reference_vs_live_real",
        metrics={"cosine_shift": 0.003, "is_alarm": False},
        started_at=_BASE_TIME,
    )
    _add_drift_run(
        db_session,
        task="drift_embedding",
        split="reference_vs_live_simulated",
        metrics={"cosine_shift": 0.6, "is_alarm": True},
        started_at=_BASE_TIME,
    )
    _add_drift_run(
        db_session,
        task="drift_prediction",
        split="reference_vs_live_real",
        metrics={"psi": 0.006, "status": "stable"},
        started_at=_BASE_TIME,
    )
    _add_drift_run(
        db_session,
        task="drift_prediction",
        split="reference_vs_live_simulated",
        metrics={"psi": 0.67, "status": "alarm"},
        started_at=_BASE_TIME,
    )

    response = client.get("/drift")

    assert response.status_code == 200
    body = response.json()
    assert body["real"]["embedding"]["metrics"]["is_alarm"] is False
    assert body["real"]["prediction"]["metrics"]["status"] == "stable"
    assert body["simulated"]["embedding"]["metrics"]["is_alarm"] is True
    assert body["simulated"]["prediction"]["metrics"]["status"] == "alarm"


def test_get_drift_returns_nulls_when_nothing_persisted(
    db_session: Session, client: TestClient
) -> None:
    response = client.get("/drift")

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "real": {"embedding": None, "prediction": None},
        "simulated": {"embedding": None, "prediction": None},
    }


def test_get_drift_returns_only_the_newest_run_per_scenario(
    db_session: Session, client: TestClient
) -> None:
    _add_drift_run(
        db_session,
        task="drift_embedding",
        split="reference_vs_live_real",
        metrics={"cosine_shift": 0.9, "is_alarm": True},
        started_at=_BASE_TIME,
    )
    _add_drift_run(
        db_session,
        task="drift_embedding",
        split="reference_vs_live_real",
        metrics={"cosine_shift": 0.003, "is_alarm": False},
        started_at=_BASE_TIME + timedelta(hours=1),
    )

    response = client.get("/drift")

    assert response.status_code == 200
    body = response.json()
    assert body["real"]["embedding"]["metrics"]["cosine_shift"] == pytest.approx(0.003)


def test_get_drift_ignores_unrelated_eval_runs(db_session: Session, client: TestClient) -> None:
    db_session.add(
        EvalRun(
            task="intent",
            model_version="baseline_intent_v1",
            dataset="bitext",
            split="test",
            metrics={"macro_f1": 0.9},
            params={},
            status="completed",
        )
    )
    db_session.commit()

    response = client.get("/drift")

    assert response.status_code == 200
    assert response.json() == {
        "real": {"embedding": None, "prediction": None},
        "simulated": {"embedding": None, "prediction": None},
    }
