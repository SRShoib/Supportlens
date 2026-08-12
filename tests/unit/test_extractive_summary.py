import pytest

from ml.inference.extractive_summary import ExtractiveSummaryPredictor


def test_takes_first_k_turns() -> None:
    predictor = ExtractiveSummaryPredictor(k=2)
    dialogue = "Customer: my order is late\nAgent: sorry about that\nCustomer: when will it arrive"

    [result] = predictor.predict([dialogue])

    assert result.summary == "Customer: my order is late Agent: sorry about that"


def test_fewer_turns_than_k_returns_all_of_them() -> None:
    predictor = ExtractiveSummaryPredictor(k=5)
    dialogue = "Customer: hello\nAgent: hi"

    [result] = predictor.predict([dialogue])

    assert result.summary == "Customer: hello Agent: hi"


def test_ignores_blank_lines() -> None:
    predictor = ExtractiveSummaryPredictor(k=2)
    dialogue = "Customer: hello\n\nAgent: hi there"

    [result] = predictor.predict([dialogue])

    assert result.summary == "Customer: hello Agent: hi there"


def test_predict_is_length_preserving() -> None:
    predictor = ExtractiveSummaryPredictor()
    results = predictor.predict(["Customer: a\nAgent: b", "Customer: c"])
    assert len(results) == 2


def test_predict_empty_texts_returns_empty_list() -> None:
    assert ExtractiveSummaryPredictor().predict([]) == []


def test_rejects_non_positive_k() -> None:
    with pytest.raises(ValueError, match="k must be"):
        ExtractiveSummaryPredictor(k=0)
