"""SPEC M9's /metrics dashboard data source: reads straight from Postgres
`eval_runs` rows persisted by every scripts/generate_*_report.py and
scripts/compute_drift.py -- this router never computes a metric itself
(CLAUDE.md rule #5; SPEC M9's accept criterion: "all metrics render from
Postgres eval runs")."""

from typing import Annotated

from fastapi import APIRouter, Query
from sqlalchemy import select

from api.db.models import EvalRun
from api.deps import DbDep
from api.schemas.eval_run import EvalRunOut

router = APIRouter(prefix="/eval-runs", tags=["eval-runs"])


@router.get("", response_model=list[EvalRunOut])
def list_eval_runs(
    db: DbDep,
    task: str | None = None,
    model_version: str | None = None,
    limit: Annotated[int, Query(le=500, ge=1)] = 200,
) -> list[EvalRun]:
    """Newest first. No `task` filter returns every task's runs interleaved
    -- the dashboard's per-task sections each call this with their own
    `task` value rather than filtering client-side."""
    stmt = select(EvalRun).order_by(EvalRun.started_at.desc()).limit(limit)
    if task is not None:
        stmt = stmt.where(EvalRun.task == task)
    if model_version is not None:
        stmt = stmt.where(EvalRun.model_version == model_version)
    return list(db.scalars(stmt).all())
