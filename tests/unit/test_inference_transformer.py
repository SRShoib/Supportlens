from pathlib import Path

import pytest

from ml.inference.transformer import TransformerPredictor

STUB_MODEL_DIR = (
    Path(__file__).resolve().parents[1] / "fixtures" / "models" / "stub_transformer_intent"
)


@pytest.fixture
def predictor() -> TransformerPredictor:
    return TransformerPredictor(STUB_MODEL_DIR)


def test_predict_returns_one_result_per_text(predictor: TransformerPredictor) -> None:
    results = predictor.predict(["cancel my order please", "where is my package"])
    assert len(results) == 2


def test_predict_empty_list_returns_empty(predictor: TransformerPredictor) -> None:
    assert predictor.predict([]) == []


def test_predict_label_is_one_of_the_trained_labels(predictor: TransformerPredictor) -> None:
    results = predictor.predict(["please cancel my order"])
    assert results[0].label in {"cancel_order", "track_order", "refund_request"}


def test_probabilities_sum_to_one(predictor: TransformerPredictor) -> None:
    results = predictor.predict(["please cancel my order"])
    assert results[0].probabilities is not None
    assert abs(sum(results[0].probabilities.values()) - 1.0) < 1e-5


def test_score_matches_max_probability(predictor: TransformerPredictor) -> None:
    results = predictor.predict(["please cancel my order"])
    assert results[0].probabilities is not None
    assert results[0].score == max(results[0].probabilities.values())


def test_missing_export_dir_raises() -> None:
    with pytest.raises(FileNotFoundError):
        TransformerPredictor(Path("does/not/exist"))
