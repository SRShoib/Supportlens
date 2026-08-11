from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from api.db.models import EvalRun
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sqlalchemy.orm import Session


class EvalMetrics(Protocol):
    """Whatever a task's metrics dataclass looks like, persist_eval_run only
    needs it to render as a JSONB-safe dict. Lets M4's span metrics
    (ml/evaluation/span_metrics.py) persist through the same function as
    classification metrics without reshaping either into the other's
    columns."""

    def to_metrics_dict(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ClassificationMetrics:
    macro_f1: float
    per_class_f1: dict[str, float]
    confusion_matrix: list[list[int]]
    labels: list[str]

    def to_metrics_dict(self) -> dict[str, Any]:
        return {
            "macro_f1": self.macro_f1,
            "per_class_f1": self.per_class_f1,
            "confusion_matrix": self.confusion_matrix,
            "labels": self.labels,
        }


def compute_classification_metrics(
    y_true: list[str], y_pred: list[str], labels: list[str]
) -> ClassificationMetrics:
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    per_class_f1 = {label: float(report[label]["f1-score"]) for label in labels}
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return ClassificationMetrics(
        macro_f1=float(macro_f1),
        per_class_f1=per_class_f1,
        confusion_matrix=cm.tolist(),
        labels=labels,
    )


def persist_eval_run(
    session: Session,
    *,
    task: str,
    model_version: str,
    dataset: str,
    split: str,
    metrics: EvalMetrics,
    params: dict[str, Any],
) -> EvalRun:
    """The single source of truth for every number in the README/dashboard
    (CLAUDE.md rule #5) — a metric that isn't persisted here is a bug."""
    eval_run = EvalRun(
        task=task,
        model_version=model_version,
        dataset=dataset,
        split=split,
        metrics=metrics.to_metrics_dict(),
        params=params,
        status="completed",
        finished_at=datetime.now(UTC),
    )
    session.add(eval_run)
    session.commit()
    return eval_run
