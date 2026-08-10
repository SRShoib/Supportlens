import pytest
from api.db import session as session_module
from api.db.session import make_engine
from api.main import app
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

pytestmark = pytest.mark.integration


def test_healthz_returns_ok(monkeypatch: pytest.MonkeyPatch, database_url: str) -> None:
    engine = make_engine(database_url)
    monkeypatch.setattr(session_module, "engine", engine)
    monkeypatch.setattr(session_module, "SessionLocal", sessionmaker(bind=engine))

    with TestClient(app) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "db": "ok", "version": "0.1.0"}
