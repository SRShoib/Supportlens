import pytest

from ml.evaluation.rouge_metrics import compute_rouge_metrics


def test_identical_prediction_and_reference_scores_near_one() -> None:
    text = "the customer requested a refund for the late order"
    metrics = compute_rouge_metrics([text], [text])

    assert metrics.rouge1 == pytest.approx(1.0)
    assert metrics.rouge2 == pytest.approx(1.0)
    assert metrics.rouge_l == pytest.approx(1.0)
    assert metrics.n == 1


def test_completely_unrelated_text_scores_zero() -> None:
    metrics = compute_rouge_metrics(["apples bananas cherries"], ["xyz qrs tuv"])

    assert metrics.rouge1 == 0.0
    assert metrics.rouge2 == 0.0
    assert metrics.rouge_l == 0.0


def test_averages_over_multiple_pairs() -> None:
    text = "the order shipped yesterday"
    metrics_perfect = compute_rouge_metrics([text, text], [text, text])
    metrics_mixed = compute_rouge_metrics([text, "totally unrelated words here"], [text, text])

    assert metrics_mixed.rouge1 < metrics_perfect.rouge1
    assert metrics_mixed.n == 2


def test_raises_on_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        compute_rouge_metrics(["a", "b"], ["a"])


def test_raises_on_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one"):
        compute_rouge_metrics([], [])


def test_to_metrics_dict_keys_are_exactly_the_persisted_columns() -> None:
    # Regression guard: scripts/generate_m6_report.py reads run.metrics["rouge1"]
    # etc. directly from persisted EvalRun rows. Any drift here silently breaks
    # that report.
    metrics = compute_rouge_metrics(["a b c"], ["a b c"])
    assert set(metrics.to_metrics_dict()) == {"rouge1", "rouge2", "rougeL", "n"}
