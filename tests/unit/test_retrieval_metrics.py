import pytest

from ml.evaluation.retrieval_metrics import compute_hit_rate, hit_at_k


def test_hit_at_k_true_when_relevant_id_is_within_top_k() -> None:
    assert hit_at_k(["a", "b", "c"], "b", k=5) is True


def test_hit_at_k_false_when_relevant_id_is_beyond_k() -> None:
    assert hit_at_k(["a", "b", "c", "d", "e", "f"], "f", k=5) is False


def test_hit_at_k_false_when_relevant_id_never_appears() -> None:
    assert hit_at_k(["a", "b"], "z", k=5) is False


def test_compute_hit_rate_averages_across_queries() -> None:
    results = [["a", "b"], ["x", "y"], ["m", "n", "z"]]
    relevant = ["a", "z", "z"]  # hit, miss, hit

    metrics = compute_hit_rate(results, relevant, k=5)

    assert metrics.hit_rate_at_k == pytest.approx(2 / 3)
    assert metrics.k == 5
    assert metrics.n_queries == 3


def test_compute_hit_rate_empty_input_returns_zero_without_dividing_by_zero() -> None:
    metrics = compute_hit_rate([], [], k=5)

    assert metrics.hit_rate_at_k == 0.0
    assert metrics.n_queries == 0


def test_compute_hit_rate_mismatched_lengths_raises() -> None:
    with pytest.raises(ValueError):
        compute_hit_rate([["a"]], [], k=5)


def test_to_metrics_dict_shape() -> None:
    metrics = compute_hit_rate([["a"]], ["a"], k=5)

    assert metrics.to_metrics_dict() == {"hit_rate_at_k": 1.0, "k": 5, "n_queries": 1}
