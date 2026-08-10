from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

from ml.inference.base import TaskResult


class BaselinePredictor:
    """Loads a joblib TF-IDF + LogisticRegression/LinearSVC pipeline once and
    serves predict() — the latency budget (<150ms, SPEC §3) rules out
    reloading the pipeline per request."""

    def __init__(self, model_path: Path) -> None:
        self._pipeline: Pipeline = joblib.load(model_path)
        classifier = self._pipeline.named_steps["clf"]
        self._supports_proba = hasattr(classifier, "predict_proba")

    def predict(self, texts: list[str]) -> list[TaskResult]:
        if not texts:
            return []

        predictions = self._pipeline.predict(texts)

        if self._supports_proba:
            proba_matrix = self._pipeline.predict_proba(texts)
            classes = self._pipeline.named_steps["clf"].classes_
            return [
                TaskResult(
                    label=str(label),
                    score=float(max(proba_row)),
                    probabilities={
                        str(c): float(p) for c, p in zip(classes, proba_row, strict=True)
                    },
                )
                for label, proba_row in zip(predictions, proba_matrix, strict=True)
            ]

        # LinearSVC etc. have no calibrated probabilities. decision_function
        # is 2D (n_samples, n_classes) for every task in this repo (intent:
        # 27 classes, urgency: 3) — both are inherently multi-class, so this
        # never hits sklearn's binary 1D-output special case.
        decision = self._pipeline.decision_function(texts)
        scores = decision.max(axis=1)
        return [
            TaskResult(label=str(label), score=float(score), probabilities=None)
            for label, score in zip(predictions, scores, strict=True)
        ]
