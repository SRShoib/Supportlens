from pydantic import BaseModel

from api.schemas.eval_run import EvalRunOut


class DriftScenarioOut(BaseModel):
    """One drift scenario's two signals (SPEC M9: embedding-distribution
    distance + prediction-distribution shift), each the latest persisted
    EvalRun for that (task, split) pair -- null when
    scripts/compute_drift.py hasn't run yet. `.metrics` on each carries the
    actual numbers (cosine_shift/is_alarm or psi/status/reference_dist/
    live_dist) per ml/evaluation/drift_metrics.py's to_metrics_dict()."""

    embedding: EvalRunOut | None
    prediction: EvalRunOut | None


class DriftOut(BaseModel):
    """SPEC M9's "a reference week and the live window" comparison, real
    (reference week vs. real recent traffic -- expected: no alarm) and
    simulated (reference week vs. an injected topically-different slice --
    expected: alarm fires) side by side."""

    real: DriftScenarioOut
    simulated: DriftScenarioOut
