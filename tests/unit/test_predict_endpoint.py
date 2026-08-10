from pathlib import Path

import pytest
from api.main import app
from api.routers import predict
from fastapi.testclient import TestClient

STUB_MODEL = (
    Path(__file__).resolve().parents[1] / "fixtures" / "models" / "stub_intent" / "model.joblib"
)

client = TestClient(app)


@pytest.fixture(autouse=True)
def _use_stub_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(predict, "_INTENT_MODEL_PATH", STUB_MODEL)
    predict._get_intent_predictor.cache_clear()
    yield
    predict._get_intent_predictor.cache_clear()


def test_predict_intent_returns_results() -> None:
    response = client.post("/predict/intent", json={"texts": ["please cancel my order"]})

    assert response.status_code == 200
    body = response.json()
    assert len(body["results"]) == 1
    assert body["results"][0]["label"] == "cancel_order"


def test_predict_intent_batches_multiple_texts() -> None:
    response = client.post(
        "/predict/intent", json={"texts": ["cancel my order", "where is my package"]}
    )

    assert response.status_code == 200
    assert len(response.json()["results"]) == 2


def test_predict_intent_rejects_empty_texts_list() -> None:
    response = client.post("/predict/intent", json={"texts": []})
    assert response.status_code == 422


def test_predict_intent_missing_model_returns_503(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(predict, "_INTENT_MODEL_PATH", Path("does/not/exist.joblib"))
    predict._get_intent_predictor.cache_clear()

    response = client.post("/predict/intent", json={"texts": ["hello"]})

    assert response.status_code == 503
