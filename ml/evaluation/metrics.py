from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from api.db.models import EvalRun
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sqlalchemy.orm import Session


@dataclass(frozen=True)
class ClassificationMetrics:
    macro_f1: float
    per_class_f1: dict[str, float]
    confusion_matrix: list[list[int]]
    labels: list[str]


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
    metrics: ClassificationMetrics,
    params: dict[str, Any],
) -> EvalRun:
    """The single source of truth for every number in the README/dashboard
    (CLAUDE.md rule #5) — a metric that isn't persisted here is a bug."""
    eval_run = EvalRun(
        task=task,
        model_version=model_version,
        dataset=dataset,
        split=split,
        metrics={
            "macro_f1": metrics.macro_f1,
            "per_class_f1": metrics.per_class_f1,
            "confusion_matrix": metrics.confusion_matrix,
            "labels": metrics.labels,
        },
        params=params,
        status="completed",
        finished_at=datetime.now(UTC),
    )
    session.add(eval_run)
    session.commit()
    return eval_run
