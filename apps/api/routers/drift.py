"""SPEC M9 drift panel: the latest real-vs-simulated drift EvalRuns,
persisted by scripts/compute_drift.py (task="drift_embedding"/
"drift_prediction", split="reference_vs_live_real"/
"reference_vs_live_simulated" -- see ml/evaluation/drift_metrics.py and
docs/decisions.md). Reads only; never recomputes drift live, same
"API only reads durably-stored eval runs" contract GET /eval-runs and
GET /topics/* already use."""

from fastapi import APIRouter
from sqlalchemy import select
from sqlalchemy.orm import Session

from api.db.models import EvalRun
from api.deps import DbDep
from api.schemas.drift import DriftOut, DriftScenarioOut

router = APIRouter(prefix="/drift", tags=["drift"])

_TASKS = ("drift_embedding", "drift_prediction")
_SPLITS = ("reference_vs_live_real", "reference_vs_live_simulated")


def _latest_drift_runs(db: Session) -> dict[tuple[str, str], EvalRun]:
    stmt = (
        select(EvalRun)
        .where(EvalRun.task.in_(_TASKS), EvalRun.split.in_(_SPLITS))
        .order_by(EvalRun.started_at.desc())
    )
    latest: dict[tuple[str, str], EvalRun] = {}
    for run in db.scalars(stmt).all():
        key = (run.task, run.split)
        latest.setdefault(key, run)  # first hit per key is the newest (already ordered desc)
    return latest


@router.get("", response_model=DriftOut)
def get_drift(db: DbDep) -> DriftOut:
    latest = _latest_drift_runs(db)
    return DriftOut(
        real=DriftScenarioOut(
            embedding=latest.get(("drift_embedding", "reference_vs_live_real")),
            prediction=latest.get(("drift_prediction", "reference_vs_live_real")),
        ),
        simulated=DriftScenarioOut(
            embedding=latest.get(("drift_embedding", "reference_vs_live_simulated")),
            prediction=latest.get(("drift_prediction", "reference_vs_live_simulated")),
        ),
    )
