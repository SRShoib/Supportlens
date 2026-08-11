import pytest

from ml.inference.base import TaskResult
from ml.inference.sentiment_trajectory import (
    URGENCY_WEIGHT,
    build_trajectory,
    compute_resolution_quality,
    signed_sentiment,
)


def test_signed_sentiment_positive_is_positive_score() -> None:
    assert signed_sentiment(TaskResult(label="positive", score=0.9)) == 0.9


def test_signed_sentiment_negative_is_negated_score() -> None:
    assert signed_sentiment(TaskResult(label="negative", score=0.8)) == -0.8


def test_signed_sentiment_neutral_is_zero_regardless_of_score() -> None:
    assert signed_sentiment(TaskResult(label="neutral", score=0.99)) == 0.0


@pytest.mark.parametrize("urgency_label", ["low", "medium", "high"])
def test_compute_resolution_quality_weights_by_urgency(urgency_label: str) -> None:
    final = TaskResult(label="positive", score=1.0)
    assert compute_resolution_quality(final, urgency_label) == URGENCY_WEIGHT[urgency_label]


def test_compute_resolution_quality_high_urgency_negative_final_scores_worst() -> None:
    high_urgency_negative = compute_resolution_quality(TaskResult("negative", 1.0), "high")
    low_urgency_negative = compute_resolution_quality(TaskResult("negative", 1.0), "low")
    assert high_urgency_negative > low_urgency_negative  # -0.33 > -1.0, i.e. "less bad"


def test_compute_resolution_quality_rejects_unknown_urgency_label() -> None:
    with pytest.raises(ValueError, match="unknown urgency label"):
        compute_resolution_quality(TaskResult("positive", 1.0), "critical")


def test_build_trajectory_sequence_covers_every_message_in_order() -> None:
    results = [TaskResult("negative", 0.9), TaskResult("neutral", 0.6), TaskResult("positive", 0.8)]
    trajectory = build_trajectory(results, [True, False, True], "medium")

    assert trajectory.sequence == ["negative", "neutral", "positive"]
    assert trajectory.scores == [-0.9, 0.0, 0.8]


def test_build_trajectory_final_customer_label_ignores_trailing_agent_message() -> None:
    # Last message overall is the agent's (positive) closing note; the last
    # *customer* message (negative) is what should drive resolution quality.
    results = [TaskResult("negative", 0.7), TaskResult("positive", 0.95)]
    trajectory = build_trajectory(results, [True, False], "low")

    assert trajectory.final_customer_label == "negative"
    assert trajectory.resolution_quality == pytest.approx(-0.7)


def test_build_trajectory_falls_back_to_last_message_when_no_customer_message() -> None:
    results = [TaskResult("positive", 0.5), TaskResult("neutral", 0.4)]
    trajectory = build_trajectory(results, [False, False], "low")

    assert trajectory.final_customer_label == "neutral"


def test_build_trajectory_single_message_ticket() -> None:
    trajectory = build_trajectory([TaskResult("positive", 1.0)], [True], "low")

    assert trajectory.sequence == ["positive"]
    assert trajectory.resolution_quality == 1.0


def test_build_trajectory_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one message"):
        build_trajectory([], [], "low")


def test_build_trajectory_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        build_trajectory([TaskResult("positive", 1.0)], [True, False], "low")


def test_trajectory_to_payload_is_json_safe_shape() -> None:
    trajectory = build_trajectory([TaskResult("positive", 1.0)], [True], "low")

    payload = trajectory.to_payload()

    assert payload == {
        "sequence": ["positive"],
        "scores": [1.0],
        "final_customer_label": "positive",
        "resolution_quality": 1.0,
    }
