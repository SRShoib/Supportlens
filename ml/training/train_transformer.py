"""Fine-tunes DistilBERT or DeBERTa-v3-small for intent or urgency
classification (SPEC M3), on the exact same seed-42 splits M2's classical
baselines used (data/splits/{task}_v1.parquet) so the comparison is
apples-to-apples.

This is a GPU training script (CLAUDE.md ground rule #3): it is written and
reviewed here, but run by a human locally, never inside Docker or the API
process. See docs/m3-how-to-run-locally.md for the full setup + invocation,
and docs/decisions.md for the exact CUDA torch install command.

Run (after `make install-training` + the CUDA torch install):
  uv run python ml/training/train_transformer.py \
      --config ml/training/configs/intent_distilbert.yaml

Smoke-test only, no real training (used to sanity-check the script itself
before handing it off for a real GPU run):
  uv run python ml/training/train_transformer.py \
      --config ml/training/configs/intent_distilbert.yaml --max-steps 5
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)

from ml.evaluation.metrics import compute_classification_metrics
from ml.training.splits import load_splits

MODELS_DIR = Path("models")


@dataclass
class TrainConfig:
    model_name: str
    task: str
    num_epochs: int = 3
    batch_size: int = 32
    eval_batch_size: int = 64
    learning_rate: float = 2e-5
    max_seq_length: int = 128
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_steps: int | None = None  # smoke-test override; None means "use num_epochs"

    @classmethod
    def from_yaml(cls, path: Path) -> "TrainConfig":
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        # PyYAML's float regex requires a decimal point, so scientific
        # notation without one (e.g. "2e-5", vs. "2.0e-5") parses as a
        # string, not a float — coerce explicitly rather than relying on
        # every config file (and future edits to one) getting the syntax
        # exactly right.
        for field in ("learning_rate", "warmup_ratio", "weight_decay"):
            if field in data:
                data[field] = float(data[field])
        return cls(**data)


def _tokenize_split(
    df: Any, tokenizer: Any, max_length: int, label_to_id: dict[str, int]
) -> Dataset:
    dataset = Dataset.from_pandas(df[["text", "label"]].reset_index(drop=True))

    def tokenize_fn(batch: dict[str, list[Any]]) -> dict[str, Any]:
        encoded = tokenizer(
            batch["text"], truncation=True, max_length=max_length, padding="max_length"
        )
        encoded["labels"] = [label_to_id[label] for label in batch["label"]]
        return encoded

    return dataset.map(tokenize_fn, batched=True, remove_columns=["text", "label"])


def _make_compute_metrics(labels: list[str]) -> Any:
    def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
        predictions = np.argmax(eval_pred.predictions, axis=1)
        y_true = [labels[i] for i in eval_pred.label_ids]
        y_pred = [labels[i] for i in predictions]
        metrics = compute_classification_metrics(y_true, y_pred, labels)
        return {"macro_f1": metrics.macro_f1}

    return compute_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Smoke-test override: cap total training steps instead of running num_epochs",
    )
    args = parser.parse_args()

    config = TrainConfig.from_yaml(args.config)
    if args.max_steps is not None:
        config.max_steps = args.max_steps

    df = load_splits(f"{config.task}_v1")
    labels = sorted(df["label"].unique())
    label_to_id = {label: i for i, label in enumerate(labels)}

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    print(
        f"task={config.task} model={config.model_name}: "
        f"{len(train_df)} train / {len(val_df)} val / {len(test_df)} test, {len(labels)} classes"
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    train_ds = _tokenize_split(train_df, tokenizer, config.max_seq_length, label_to_id)
    val_ds = _tokenize_split(val_df, tokenizer, config.max_seq_length, label_to_id)
    test_ds = _tokenize_split(test_df, tokenizer, config.max_seq_length, label_to_id)

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=len(labels),
        id2label=dict(enumerate(labels)),
        label2id=label_to_id,
    )

    model_slug = config.model_name.split("/")[-1]
    output_dir = MODELS_DIR / f"transformer_{config.task}_{model_slug}_v1"

    # TrainingArguments in this transformers version only accepts
    # warmup_steps, not warmup_ratio (dropped at some point after 4.46) —
    # computed manually so the config's ratio still means what it says
    # regardless of dataset size or epoch count.
    steps_per_epoch = max(len(train_df) // config.batch_size, 1)
    total_steps = config.max_steps or steps_per_epoch * config.num_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=config.num_epochs,
        max_steps=config.max_steps or -1,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        learning_rate=config.learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=config.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=_make_compute_metrics(labels),
    )

    trainer.train()

    test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
    print(f"test macro_f1: {test_metrics['test_macro_f1']:.4f}")

    export_dir = output_dir / "final"
    trainer.save_model(str(export_dir))
    tokenizer.save_pretrained(str(export_dir))
    (export_dir / "label_map.json").write_text(
        json.dumps({"labels": labels}, indent=2), encoding="utf-8"
    )

    print(f"exported: {export_dir}")


if __name__ == "__main__":
    main()
