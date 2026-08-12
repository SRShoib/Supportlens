"""Drift detection (SPEC M9: "embedding-distribution distance (e.g. MMD or
centroid cosine shift) + prediction-distribution shift (PSI) between a
reference week and the live window"). Pure, DB-free -- callers assemble the
per-entity vectors/label-counts from Postgres + the M7 embeddings artifact
(scripts/compute_drift.py), this module only does the math, same split as
trend_metrics.py/topic_metrics.py.

Two signals, two different threshold stories:

- PSI's 0.1 (watch) / 0.25 (alarm) bands are the standard, already
  -established interpretation of the metric (not corpus-specific) -- no
  reason to invent a different one.
- Centroid cosine shift has no such external convention, so its threshold
  was measured against this repo's real corpus, the same way M7's z-score
  and M8's MIN_CONFIDENCE were: a real week-vs-week comparison (no injected
  drift) measures ~0.004-0.005; the M9 simulated scenario (a Bitext slice
  injected as the "live window" -- see docs/decisions.md and
  scripts/compute_drift.py) measures ~0.64. EMBEDDING_DRIFT_THRESHOLD sits
  an order of magnitude above the former and well below the latter.
"""

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

EMBEDDING_DRIFT_THRESHOLD = 0.05
"""Centroid cosine shift alarm gate -- see module docstring for the
real-corpus measurement that set this."""

PSI_WATCH_THRESHOLD = 0.1
PSI_ALARM_THRESHOLD = 0.25
"""Standard PSI interpretation bands: <0.1 stable, 0.1-0.25 watch, >0.25
alarm. Confirmed against this repo's real urgency-label distribution too
(docs/decisions.md): real week-vs-week PSI measures ~0.004-0.01
(comfortably "stable"), the simulated scenario measures ~0.67 (comfortably
"alarm")."""

_PSI_EPSILON = 1e-6


def centroid_cosine_shift(reference_vectors: np.ndarray, live_vectors: np.ndarray) -> float:
    """1 - cosine_similarity(mean(reference), mean(live)). 0 means the two
    centroids point in exactly the same direction (no shift); larger is
    more drift. Requires at least one vector on each side."""
    if len(reference_vectors) == 0 or len(live_vectors) == 0:
        raise ValueError("centroid_cosine_shift requires at least one vector in each set")

    reference_centroid = np.asarray(reference_vectors).mean(axis=0)
    live_centroid = np.asarray(live_vectors).mean(axis=0)
    denom = np.linalg.norm(reference_centroid) * np.linalg.norm(live_centroid)
    if denom == 0:
        raise ValueError("centroid_cosine_shift requires non-zero centroid vectors")
    cosine_similarity = float(np.dot(reference_centroid, live_centroid) / denom)
    return 1.0 - cosine_similarity


@dataclass(frozen=True)
class EmbeddingDriftResult:
    cosine_shift: float
    is_alarm: bool
    reference_n: int
    live_n: int

    def to_metrics_dict(self) -> dict[str, Any]:
        return {
            "cosine_shift": self.cosine_shift,
            "is_alarm": self.is_alarm,
            "reference_n": self.reference_n,
            "live_n": self.live_n,
            "threshold": EMBEDDING_DRIFT_THRESHOLD,
        }


def compute_embedding_drift(
    reference_vectors: np.ndarray, live_vectors: np.ndarray
) -> EmbeddingDriftResult:
    shift = centroid_cosine_shift(reference_vectors, live_vectors)
    return EmbeddingDriftResult(
        cosine_shift=shift,
        is_alarm=shift > EMBEDDING_DRIFT_THRESHOLD,
        reference_n=len(reference_vectors),
        live_n=len(live_vectors),
    )


def population_stability_index(
    reference_counts: dict[str, int], live_counts: dict[str, int]
) -> float:
    """Standard PSI: sum over bins of (live% - ref%) * ln(live% / ref%).
    Zero-count bins are additive-smoothed by _PSI_EPSILON (same pattern as
    topic_metrics.py's NPMI) rather than skipped, so a label that's present
    in one distribution and entirely absent from the other still
    contributes a real (large) number instead of silently dropping out."""
    labels = sorted(set(reference_counts) | set(live_counts))
    total_reference = sum(reference_counts.values())
    total_live = sum(live_counts.values())
    if total_reference == 0 or total_live == 0:
        raise ValueError("population_stability_index requires at least one observation per side")

    psi = 0.0
    for label in labels:
        p_reference = max(reference_counts.get(label, 0) / total_reference, _PSI_EPSILON)
        p_live = max(live_counts.get(label, 0) / total_live, _PSI_EPSILON)
        psi += (p_live - p_reference) * math.log(p_live / p_reference)
    return psi


def _psi_status(psi: float) -> str:
    if psi > PSI_ALARM_THRESHOLD:
        return "alarm"
    if psi > PSI_WATCH_THRESHOLD:
        return "watch"
    return "stable"


@dataclass(frozen=True)
class PredictionDriftResult:
    psi: float
    status: str  # "stable" | "watch" | "alarm"
    reference_dist: dict[str, float]  # normalized proportions
    live_dist: dict[str, float]
    reference_n: int
    live_n: int

    def to_metrics_dict(self) -> dict[str, Any]:
        return {
            "psi": self.psi,
            "status": self.status,
            "reference_dist": self.reference_dist,
            "live_dist": self.live_dist,
            "reference_n": self.reference_n,
            "live_n": self.live_n,
            "watch_threshold": PSI_WATCH_THRESHOLD,
            "alarm_threshold": PSI_ALARM_THRESHOLD,
        }


def compute_prediction_drift(
    reference_counts: dict[str, int], live_counts: dict[str, int]
) -> PredictionDriftResult:
    psi = population_stability_index(reference_counts, live_counts)
    total_reference = sum(reference_counts.values())
    total_live = sum(live_counts.values())
    return PredictionDriftResult(
        psi=psi,
        status=_psi_status(psi),
        reference_dist={k: v / total_reference for k, v in reference_counts.items()},
        live_dist={k: v / total_live for k, v in live_counts.items()},
        reference_n=total_reference,
        live_n=total_live,
    )
