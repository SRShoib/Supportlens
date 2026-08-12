import numpy as np
import pytest

from ml.evaluation.drift_metrics import (
    EMBEDDING_DRIFT_THRESHOLD,
    PSI_ALARM_THRESHOLD,
    PSI_WATCH_THRESHOLD,
    centroid_cosine_shift,
    compute_embedding_drift,
    compute_prediction_drift,
    population_stability_index,
)


def test_centroid_cosine_shift_is_zero_for_identical_centroids() -> None:
    vectors = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    assert centroid_cosine_shift(vectors, vectors) == pytest.approx(0.0, abs=1e-9)


def test_centroid_cosine_shift_is_two_for_opposite_centroids() -> None:
    reference = np.array([[1.0, 0.0]])
    live = np.array([[-1.0, 0.0]])
    assert centroid_cosine_shift(reference, live) == pytest.approx(2.0)


def test_centroid_cosine_shift_scale_invariant() -> None:
    reference = np.array([[1.0, 0.0]])
    live_same_direction = np.array([[5.0, 0.0]])
    assert centroid_cosine_shift(reference, live_same_direction) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("empty_side", ["reference", "live"])
def test_centroid_cosine_shift_requires_nonempty_vectors(empty_side: str) -> None:
    vectors = np.array([[1.0, 0.0]])
    empty = np.empty((0, 2))
    with pytest.raises(ValueError):
        centroid_cosine_shift(
            empty if empty_side == "reference" else vectors,
            vectors if empty_side == "reference" else empty,
        )


def test_compute_embedding_drift_no_alarm_on_small_real_world_scale_shift() -> None:
    """Real week-vs-week noise on this repo's corpus measures ~0.004-0.005
    (docs/decisions.md) -- a shift at that scale must not fire."""
    rng = np.random.default_rng(42)
    reference = rng.normal(loc=0.0, scale=1.0, size=(200, 32))
    live = reference + rng.normal(loc=0.0, scale=0.02, size=(200, 32))

    result = compute_embedding_drift(reference, live)

    assert result.cosine_shift < EMBEDDING_DRIFT_THRESHOLD
    assert result.is_alarm is False
    assert result.reference_n == 200
    assert result.live_n == 200


def test_compute_embedding_drift_alarms_on_a_clearly_shifted_distribution() -> None:
    """The M9 simulated (topically-different, Bitext-injected) scenario
    measures ~0.64 on the real corpus -- an orthogonal-ish centroid shift
    at that scale must fire."""
    reference = np.array([[1.0, 0.0, 0.0]] * 50)
    live = np.array([[0.0, 1.0, 0.0]] * 50)

    result = compute_embedding_drift(reference, live)

    assert result.cosine_shift > EMBEDDING_DRIFT_THRESHOLD
    assert result.is_alarm is True


def test_population_stability_index_is_zero_for_identical_distributions() -> None:
    dist = {"low": 700, "medium": 200, "high": 100}
    assert population_stability_index(dist, dist) == pytest.approx(0.0, abs=1e-9)


def test_population_stability_index_is_symmetric_in_magnitude_not_sign() -> None:
    # PSI itself isn't symmetric term-by-term, but swapping reference/live
    # on a two-bin distribution should still read as "some real shift" both
    # ways, not zero one way and nonzero the other.
    a = {"x": 90, "y": 10}
    b = {"x": 10, "y": 90}
    assert population_stability_index(a, b) > PSI_ALARM_THRESHOLD
    assert population_stability_index(b, a) > PSI_ALARM_THRESHOLD


def test_population_stability_index_requires_nonempty_counts() -> None:
    with pytest.raises(ValueError):
        population_stability_index({}, {"low": 1})
    with pytest.raises(ValueError):
        population_stability_index({"low": 1}, {})


def test_compute_prediction_drift_stable_on_real_world_scale_weekly_noise() -> None:
    """Real week-vs-week urgency-label PSI on this repo's corpus measures
    ~0.004-0.01 (docs/decisions.md) -- must land in the "stable" band."""
    reference = {"low": 2602, "medium": 723, "high": 296}
    live = {"low": 2386, "medium": 826, "high": 330}

    result = compute_prediction_drift(reference, live)

    assert result.psi < PSI_WATCH_THRESHOLD
    assert result.status == "stable"
    assert result.reference_n == 3621
    assert result.live_n == 3542
    assert result.reference_dist["low"] == pytest.approx(2602 / 3621)


def test_compute_prediction_drift_alarms_on_the_simulated_bitext_style_skew() -> None:
    """The M9 simulated scenario's real urgency-label skew (mostly "low")
    measures PSI ~0.67 against the same reference week -- must alarm."""
    reference = {"low": 2602, "medium": 723, "high": 296}
    live = {"low": 25571, "medium": 1282, "high": 19}

    result = compute_prediction_drift(reference, live)

    assert result.psi > PSI_ALARM_THRESHOLD
    assert result.status == "alarm"


def test_compute_prediction_drift_watch_band_between_thresholds() -> None:
    reference = {"low": 800, "medium": 150, "high": 50}
    live = {"low": 650, "medium": 230, "high": 120}

    result = compute_prediction_drift(reference, live)

    assert PSI_WATCH_THRESHOLD < result.psi <= PSI_ALARM_THRESHOLD
    assert result.status == "watch"


def test_compute_prediction_drift_missing_label_on_one_side_still_scores() -> None:
    reference = {"low": 100, "medium": 100}
    live = {"low": 100, "medium": 100, "high": 50}

    result = compute_prediction_drift(reference, live)

    assert result.reference_dist.get("high") is None
    assert result.live_dist["high"] == pytest.approx(50 / 250)
    assert result.psi > 0
