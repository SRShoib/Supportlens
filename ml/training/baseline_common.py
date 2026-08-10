"""Shared TF-IDF (word + char n-gram) + LogisticRegression/LinearSVC training
and evaluation logic for M2's classical baselines (SPEC M2)."""

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from ml.evaluation.metrics import ClassificationMetrics, compute_classification_metrics

MODELS_DIR = Path("models")


def build_feature_pipeline() -> FeatureUnion:
    return FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30_000)),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=30_000
                ),
            ),
        ]
    )


@dataclass(frozen=True)
class TrainedVariant:
    name: str
    pipeline: Pipeline
    val_metrics: ClassificationMetrics


def train_variants(
    train_df: pd.DataFrame, val_df: pd.DataFrame, labels: list[str]
) -> list[TrainedVariant]:
    variants = []
    classifiers: list[tuple[str, LogisticRegression | LinearSVC]] = [
        ("logistic_regression", LogisticRegression(class_weight="balanced", max_iter=1000)),
        ("linear_svc", LinearSVC(class_weight="balanced")),
    ]
    for name, classifier in classifiers:
        pipeline = Pipeline([("features", build_feature_pipeline()), ("clf", classifier)])
        pipeline.fit(train_df["text"], train_df["label"])
        y_pred = pipeline.predict(val_df["text"])
        metrics = compute_classification_metrics(val_df["label"].tolist(), list(y_pred), labels)
        variants.append(TrainedVariant(name=name, pipeline=pipeline, val_metrics=metrics))
        print(f"  {name}: val macro_f1={metrics.macro_f1:.4f}")
    return variants


def pick_best(variants: list[TrainedVariant]) -> TrainedVariant:
    return max(variants, key=lambda v: v.val_metrics.macro_f1)


def evaluate_on_test(
    pipeline: Pipeline, test_df: pd.DataFrame, labels: list[str]
) -> ClassificationMetrics:
    y_pred = pipeline.predict(test_df["text"])
    return compute_classification_metrics(test_df["label"].tolist(), list(y_pred), labels)


def export_model(task: str, pipeline: Pipeline, version: str = "v1") -> Path:
    out_dir = MODELS_DIR / f"baseline_{task}_{version}"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.joblib"
    joblib.dump(pipeline, model_path)
    return model_path
