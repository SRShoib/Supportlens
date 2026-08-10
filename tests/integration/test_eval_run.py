import pytest
from api.db.models import EvalRun
from sqlalchemy import select
from sqlalchemy.orm import Session

from ml.evaluation.metrics import compute_classification_metrics, persist_eval_run

pytestmark = pytest.mark.integration


def test_persist_eval_run_roundtrip(db_session: Session) -> None:
    metrics = compute_classification_metrics(
        y_true=["a", "b", "a", "b"], y_pred=["a", "b", "a", "a"], labels=["a", "b"]
    )

    persist_eval_run(
        db_session,
        task="intent",
        model_version="baseline_linear_svc_v1",
        dataset="bitext",
        split="test",
        metrics=metrics,
        params={"seed": 42},
    )

    stored = db_session.scalars(select(EvalRun).where(EvalRun.task == "intent")).one()
    assert stored.model_version == "baseline_linear_svc_v1"
    assert stored.dataset == "bitext"
    assert stored.split == "test"
    assert stored.status == "completed"
    assert stored.finished_at is not None
    assert stored.metrics["macro_f1"] == metrics.macro_f1
    assert stored.metrics["labels"] == ["a", "b"]
    assert stored.params == {"seed": 42}


def test_multiple_eval_runs_for_same_task_all_persist(db_session: Session) -> None:
    metrics = compute_classification_metrics(["a"], ["a"], labels=["a"])

    for model_version in ("baseline_logistic_regression_v1", "baseline_linear_svc_v1"):
        persist_eval_run(
            db_session,
            task="urgency",
            model_version=model_version,
            dataset="twitter_slice_v1",
            split="val",
            metrics=metrics,
            params={},
        )

    stored = db_session.scalars(select(EvalRun).where(EvalRun.task == "urgency")).all()
    assert {r.model_version for r in stored} == {
        "baseline_logistic_regression_v1",
        "baseline_linear_svc_v1",
    }
