from pathlib import Path

import pytest

from ml.inference.baseline import BaselinePredictor

STUB_MODEL = (
    Path(__file__).resolve().parents[1] / "fixtures" / "models" / "stub_intent" / "model.joblib"
)


@pytest.fixture
def predictor() -> BaselinePredictor:
    return BaselinePredictor(STUB_MODEL)


def test_predict_returns_one_result_per_text(predictor: BaselinePredictor) -> None:
    results = predictor.predict(["cancel my order please", "where is my package"])
    assert len(results) == 2


def test_predict_labels_are_plausible(predictor: BaselinePredictor) -> None:
    results = predictor.predict(["please cancel my order"])
    assert results[0].label == "cancel_order"


def test_predict_empty_list_returns_empty(predictor: BaselinePredictor) -> None:
    assert predictor.predict([]) == []


def test_logistic_regression_stub_has_probabilities(predictor: BaselinePredictor) -> None:
    # the stub fixture is a LogisticRegression pipeline, which supports
    # predict_proba unlike LinearSVC
    results = predictor.predict(["I need a refund"])
    assert results[0].probabilities is not None
    assert abs(sum(results[0].probabilities.values()) - 1.0) < 1e-6


def test_score_matches_max_probability(predictor: BaselinePredictor) -> None:
    results = predictor.predict(["track my order status"])
    assert results[0].probabilities is not None
    assert results[0].score == max(results[0].probabilities.values())


def test_missing_model_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        BaselinePredictor(Path("does/not/exist.joblib"))
