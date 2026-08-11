import pytest
from api.db.models import EvalRun
from sqlalchemy import select
from sqlalchemy.orm import Session

from ml.data.ner.schema import CharSpan
from ml.evaluation.metrics import persist_eval_run
from ml.evaluation.span_metrics import compute_span_metrics

pytestmark = pytest.mark.integration


def test_span_metrics_persist_through_the_same_persist_eval_run(db_session: Session) -> None:
    # Proves the EvalMetrics seam works against real Postgres: SpanMetrics
    # is a completely different shape from ClassificationMetrics, but both
    # persist through the identical function because both implement
    # to_metrics_dict().
    gold = [
        [CharSpan(6, 15, "ORDER_ID", "ORD-99321"), CharSpan(24, 33, "DATE", "yesterday")],
        [CharSpan(8, 14, "AMOUNT", "$49.99")],
    ]
    pred = [
        [CharSpan(6, 15, "ORDER_ID", "ORD-99321")],
        [CharSpan(8, 14, "AMOUNT", "$49.99")],
    ]
    labels = ["ORDER_ID", "PRODUCT", "DATE", "AMOUNT", "ACCOUNT_REF"]
    metrics = compute_span_metrics(gold, pred, labels)

    persist_eval_run(
        db_session,
        task="entities",
        model_version="rules_ner_v1",
        dataset="ner_gold_v1",
        split="gold",
        metrics=metrics,
        params={"n_documents": metrics.n_documents},
    )

    stored = db_session.scalars(select(EvalRun).where(EvalRun.task == "entities")).one()
    assert stored.model_version == "rules_ner_v1"
    assert stored.dataset == "ner_gold_v1"
    assert stored.split == "gold"
    assert stored.status == "completed"
    assert stored.finished_at is not None

    # Round-tripped through JSONB -- per-type P/R/F1 for every label survive,
    # not just the top-line micro/macro numbers.
    assert stored.metrics["micro_f1"] == metrics.micro_f1
    assert stored.metrics["boundary_f1"] == metrics.boundary_f1
    assert stored.metrics["partial_f1"] == metrics.partial_f1
    assert stored.metrics["per_type"]["ORDER_ID"]["f1"] == metrics.per_type["ORDER_ID"].f1
    assert stored.metrics["per_type"]["DATE"]["fn"] == metrics.per_type["DATE"].fn
    assert set(stored.metrics["per_type"]) == set(labels)


def test_classification_and_span_eval_runs_coexist_for_different_tasks(
    db_session: Session,
) -> None:
    from ml.evaluation.metrics import compute_classification_metrics

    classification_metrics = compute_classification_metrics(
        y_true=["a", "b"], y_pred=["a", "b"], labels=["a", "b"]
    )
    persist_eval_run(
        db_session,
        task="intent",
        model_version="baseline_linear_svc_v1",
        dataset="bitext",
        split="test",
        metrics=classification_metrics,
        params={},
    )

    span_metrics = compute_span_metrics([[]], [[]], ["ORDER_ID"])
    persist_eval_run(
        db_session,
        task="entities",
        model_version="rules_ner_v1",
        dataset="ner_gold_v1",
        split="gold",
        metrics=span_metrics,
        params={},
    )

    rows = db_session.scalars(select(EvalRun)).all()
    tasks = {row.task for row in rows}
    assert tasks == {"intent", "entities"}
