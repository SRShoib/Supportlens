import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ml.inference.base import TaskResult


class TransformerPredictor:
    """Loads an exported HF sequence-classification model + tokenizer once and
    serves predict() on CPU (SPEC §3: serving is CPU-only by design). Reads
    label_map.json (written by ml/training/train_transformer.py) rather than
    parsing the model config's id2label, to keep label ordering an explicit,
    independently-checkable artifact instead of an HF-internal detail."""

    def __init__(self, export_dir: Path, max_length: int = 128) -> None:
        if not (export_dir / "model.safetensors").exists():
            raise FileNotFoundError(f"no exported model at {export_dir}")

        self._tokenizer = AutoTokenizer.from_pretrained(str(export_dir))
        self._model = AutoModelForSequenceClassification.from_pretrained(str(export_dir))
        self._model.eval()
        self._max_length = max_length
        self._labels: list[str] = json.loads(
            (export_dir / "label_map.json").read_text(encoding="utf-8")
        )["labels"]

    def predict(self, texts: list[str]) -> list[TaskResult]:
        if not texts:
            return []

        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self._max_length,
        )
        with torch.no_grad():
            logits = self._model(**inputs).logits
        probabilities = torch.softmax(logits, dim=1)

        results = []
        for row in probabilities:
            probs = {label: float(p) for label, p in zip(self._labels, row.tolist(), strict=True)}
            best_label = max(probs, key=lambda label: probs[label])
            results.append(
                TaskResult(label=best_label, score=probs[best_label], probabilities=probs)
            )
        return results
