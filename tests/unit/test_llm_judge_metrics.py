import pytest

from ml.evaluation.llm_judge_metrics import aggregate_judge_scores


def test_averages_faithfulness_and_coverage() -> None:
    rows = [
        {"faithfulness": 5, "coverage": 3, "parsed_ok": True},
        {"faithfulness": 3, "coverage": 5, "parsed_ok": True},
    ]
    metrics = aggregate_judge_scores(rows)  # type: ignore[arg-type]

    assert metrics.n == 2
    assert metrics.mean_faithfulness == 4.0
    assert metrics.mean_coverage == 4.0


def test_parsed_ok_rate_reflects_fallback_rows() -> None:
    rows = [
        {"faithfulness": 5, "coverage": 5, "parsed_ok": True},
        {"faithfulness": 3, "coverage": 3, "parsed_ok": False},
        {"faithfulness": 3, "coverage": 3, "parsed_ok": False},
    ]
    metrics = aggregate_judge_scores(rows)  # type: ignore[arg-type]

    assert metrics.parsed_ok_rate == pytest.approx(1 / 3)


def test_raises_on_empty_rows() -> None:
    with pytest.raises(ValueError, match="at least one row"):
        aggregate_judge_scores([])


def test_to_metrics_dict_keys_are_exactly_the_persisted_columns() -> None:
    metrics = aggregate_judge_scores(
        [{"faithfulness": 4, "coverage": 4, "parsed_ok": True}]  # type: ignore[list-item]
    )
    assert set(metrics.to_metrics_dict()) == {
        "n",
        "mean_faithfulness",
        "mean_coverage",
        "parsed_ok_rate",
    }
