"""Trains TF-IDF + LogisticRegression/LinearSVC baselines for intent
classification (SPEC M2, accept: macro-F1 >= 0.85 on Bitext test).

Run: uv run python -m ml.training.train_baseline_intent
"""

from api.config import get_settings
from api.db.session import SessionLocal

from ml.evaluation.metrics import persist_eval_run
from ml.training.baseline_common import evaluate_on_test, export_model, pick_best, train_variants
from ml.training.splits import load_splits

TASK = "intent"
DATASET = "bitext"


def main() -> None:
    settings = get_settings()
    df = load_splits(f"{TASK}_v1")
    labels = sorted(df["label"].unique())
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    print(
        f"training {TASK} baselines: {len(train_df)} train / {len(val_df)} val / "
        f"{len(test_df)} test, {len(labels)} classes"
    )
    variants = train_variants(train_df, val_df, labels)
    best = pick_best(variants)
    print(f"best variant: {best.name} (val macro_f1={best.val_metrics.macro_f1:.4f})")

    test_metrics = evaluate_on_test(best.pipeline, test_df, labels)
    print(f"test macro_f1={test_metrics.macro_f1:.4f}")

    model_path = export_model(TASK, best.pipeline)
    print(f"exported: {model_path}")

    session = SessionLocal()
    try:
        for variant in variants:
            persist_eval_run(
                session,
                task=TASK,
                model_version=f"baseline_{variant.name}_v1",
                dataset=DATASET,
                split="val",
                metrics=variant.val_metrics,
                params={"seed": settings.random_seed, "n_train": len(train_df)},
            )
        persist_eval_run(
            session,
            task=TASK,
            model_version=f"baseline_{best.name}_v1",
            dataset=DATASET,
            split="test",
            metrics=test_metrics,
            params={"seed": settings.random_seed, "n_train": len(train_df), "selected": True},
        )
    finally:
        session.close()


if __name__ == "__main__":
    main()
