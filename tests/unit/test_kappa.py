import pytest

from ml.evaluation.kappa import compute_kappa


def test_perfect_agreement_gives_kappa_one() -> None:
    labels = ["low", "medium", "high", "low", "medium"]
    result = compute_kappa(labels, labels)
    assert result.kappa == 1.0
    assert result.n == 5


def test_no_agreement_gives_low_kappa() -> None:
    weak = ["low", "low", "low", "low"]
    llm = ["high", "high", "high", "high"]
    result = compute_kappa(weak, llm)
    assert result.kappa <= 0.0


def test_partial_agreement_between_zero_and_one() -> None:
    weak = ["low", "medium", "high", "low", "medium", "high"]
    llm = ["low", "medium", "medium", "low", "high", "high"]
    result = compute_kappa(weak, llm)
    assert 0.0 < result.kappa < 1.0


def test_mismatched_lengths_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        compute_kappa(["low"], ["low", "high"])


def test_empty_input_returns_zero_kappa_and_count() -> None:
    result = compute_kappa([], [])
    assert result.kappa == 0.0
    assert result.n == 0
