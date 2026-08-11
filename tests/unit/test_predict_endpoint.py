from pathlib import Path

import pytest
from api.main import app
from api.routers import predict
from fastapi.testclient import TestClient

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "models"
STUB_BASELINE_MODEL = FIXTURES / "stub_intent" / "model.joblib"
STUB_TRANSFORMER_DIR = FIXTURES / "stub_transformer_intent"
STUB_NER_DIR = FIXTURES / "stub_ner"

client = TestClient(app)


@pytest.fixture(autouse=True)
def _use_stub_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(predict._BASELINE_MODEL_PATHS, "intent", STUB_BASELINE_MODEL)
    monkeypatch.setitem(predict._TRANSFORMER_MODEL_DIRS, "intent", STUB_TRANSFORMER_DIR)
    monkeypatch.setitem(predict._NER_MODEL_DIRS, "entities", STUB_NER_DIR)
    predict._get_baseline_predictor.cache_clear()
    predict._get_transformer_predictor.cache_clear()
    predict._get_ner_predictor.cache_clear()
    yield
    predict._get_baseline_predictor.cache_clear()
    predict._get_transformer_predictor.cache_clear()
    predict._get_ner_predictor.cache_clear()


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


def test_predict_intent_missing_baseline_model_returns_503(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(predict._BASELINE_MODEL_PATHS, "intent", Path("does/not/exist.joblib"))
    predict._get_baseline_predictor.cache_clear()

    response = client.post("/predict/intent", json={"texts": ["hello"]})

    assert response.status_code == 503


def test_predict_intent_defaults_to_baseline_model() -> None:
    response = client.post("/predict/intent", json={"texts": ["please cancel my order"]})
    assert response.status_code == 200


def test_predict_intent_transformer_flag_routes_to_transformer() -> None:
    response = client.post(
        "/predict/intent", json={"texts": ["please cancel my order"], "model": "transformer"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["results"][0]["label"] in {"cancel_order", "track_order", "refund_request"}


def test_predict_intent_missing_transformer_model_returns_503(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(predict._TRANSFORMER_MODEL_DIRS, "intent", Path("does/not/exist"))
    predict._get_transformer_predictor.cache_clear()

    response = client.post("/predict/intent", json={"texts": ["hello"], "model": "transformer"})

    assert response.status_code == 503


def test_predict_intent_rejects_unknown_model_flag() -> None:
    response = client.post(
        "/predict/intent", json={"texts": ["hello"], "model": "not-a-real-model"}
    )
    assert response.status_code == 422


def test_predict_urgency_endpoint_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    # urgency shares the same baseline-loading code path, keyed by task —
    # stub it independently of intent to prove the two tasks don't share
    # cached state.
    monkeypatch.setitem(predict._BASELINE_MODEL_PATHS, "urgency", STUB_BASELINE_MODEL)
    predict._get_baseline_predictor.cache_clear()

    response = client.post("/predict/urgency", json={"texts": ["hello", "urgent refund now"]})

    assert response.status_code == 200
    assert len(response.json()["results"]) == 2


def test_predict_urgency_missing_model_returns_503(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(predict._BASELINE_MODEL_PATHS, "urgency", Path("does/not/exist.joblib"))
    predict._get_baseline_predictor.cache_clear()

    response = client.post("/predict/urgency", json={"texts": ["hello"]})

    assert response.status_code == 503


def test_predict_entities_returns_results_with_baseline_default() -> None:
    response = client.post(
        "/predict/entities", json={"texts": ["order ORD-99321 shipped yesterday"]}
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["results"]) == 1
    labels = {span["label"] for span in body["results"][0]["entities"]}
    assert labels == {"ORDER_ID", "DATE"}


def test_predict_entities_span_offsets_match_the_returned_text() -> None:
    text = "order ORD-99321 shipped yesterday"
    response = client.post("/predict/entities", json={"texts": [text]})

    body = response.json()
    for span in body["results"][0]["entities"]:
        assert span["text"] == text[span["start"] : span["end"]]


def test_predict_entities_batches_multiple_texts() -> None:
    response = client.post(
        "/predict/entities",
        json={"texts": ["charged $49.99 today", "no entities in this one"]},
    )

    assert response.status_code == 200
    assert len(response.json()["results"]) == 2


def test_predict_entities_rejects_empty_texts_list() -> None:
    response = client.post("/predict/entities", json={"texts": []})
    assert response.status_code == 422


def test_predict_entities_rejects_unknown_model_flag() -> None:
    response = client.post(
        "/predict/entities", json={"texts": ["hello"], "model": "not-a-real-model"}
    )
    assert response.status_code == 422


def test_predict_entities_transformer_flag_routes_to_transformer() -> None:
    response = client.post(
        "/predict/entities",
        json={"texts": ["order shipped yesterday"], "model": "transformer"},
    )

    assert response.status_code == 200
    body = response.json()
    assert "entities" in body["results"][0]
    assert "truncated" in body["results"][0]


def test_predict_entities_missing_transformer_model_returns_503(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(predict._NER_MODEL_DIRS, "entities", Path("does/not/exist"))
    predict._get_ner_predictor.cache_clear()

    response = client.post("/predict/entities", json={"texts": ["hello"], "model": "transformer"})

    assert response.status_code == 503


def test_predict_entities_baseline_never_503s_even_when_transformer_dir_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The asymmetry this endpoint has and /predict/intent doesn't: the
    # rules extractor has no file on disk to be missing, so model="baseline"
    # always succeeds regardless of whether a real NER export exists.
    monkeypatch.setitem(predict._NER_MODEL_DIRS, "entities", Path("does/not/exist"))
    predict._get_ner_predictor.cache_clear()

    response = client.post(
        "/predict/entities", json={"texts": ["charged $49.99 today"], "model": "baseline"}
    )

    assert response.status_code == 200


def test_predict_entities_defaults_to_baseline_model() -> None:
    response = client.post("/predict/entities", json={"texts": ["charged $49.99 today"]})
    assert response.status_code == 200
