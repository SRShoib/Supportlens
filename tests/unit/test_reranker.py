import pytest

from ml.inference.reranker import rerank_by_score


def test_sorts_items_by_score_descending() -> None:
    items = ["low", "high", "mid"]
    scores = [0.1, 0.9, 0.5]

    assert rerank_by_score(items, scores) == ["high", "mid", "low"]


def test_stable_for_equal_scores() -> None:
    items = ["a", "b", "c"]
    scores = [0.5, 0.5, 0.9]

    assert rerank_by_score(items, scores) == ["c", "a", "b"]


def test_empty_input_returns_empty_list() -> None:
    assert rerank_by_score([], []) == []


def test_mismatched_lengths_raises() -> None:
    with pytest.raises(ValueError):
        rerank_by_score(["a", "b"], [0.1])
