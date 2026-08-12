"""Fine-tunes FLAN-T5-small for thread summarization (SPEC M6), on the
pooled train splits of samsum + dialogsum (data/splits/{samsum,dialogsum}_v1.parquet,
built by ml/training/summarization_data.py) -- per docs/decisions.md, both
sources' train rows are pooled into one fine-tune (more style diversity,
dialogsum's service-style dialogues are closer to this project's domain than
samsum's messenger chats), but ROUGE is only ever reported per-dataset
against each one's own test split (scripts/generate_m6_report.py), to stay
comparable to published per-benchmark numbers -- the same logic M5 applied
to tweet_eval's fixed splits.

Deliberately never touches either dataset's test split -- only train/val.
The one authoritative test-split ROUGE number (per dataset) is computed once,
by scripts/generate_m6_report.py, against the exported checkpoint -- not
reused from anything printed here.

This is a GPU training script (CLAUDE.md ground rule #3): written and
reviewed here, run by a human locally, never inside Docker or the API
process. See docs/m6-how-to-run-locally.md.

Run (after `make install-training` + the CUDA torch install):
  uv run python ml/training/train_summarization.py \
      --config ml/training/configs/thread_summary_flan_t5_small.yaml

Smoke-test only, no real training:
  uv run python ml/training/train_summarization.py \
      --config ml/training/configs/thread_summary_flan_t5_small.yaml --max-steps 5
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EvalPrediction,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from ml.evaluation.rouge_metrics import compute_rouge_metrics
from ml.inference.summarization import PROMPT_TEMPLATE
from ml.training.splits import load_splits

MODELS_DIR = Path("models")
SOURCE_DATASETS = ("samsum_v1", "dialogsum_v1")


@dataclass
class TrainConfig:
    model_name: str
    num_epochs: int = 3
    batch_size: int = 16
    eval_batch_size: int = 32
    learning_rate: float = 3e-4
    max_source_length: int = 512
    max_target_length: int = 64
    num_beams: int = 4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_steps: int | None = None  # smoke-test override; None means "use num_epochs"

    @classmethod
    def from_yaml(cls, path: Path) -> "TrainConfig":
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        # Same PyYAML scientific-notation gotcha train_transformer.py works
        # around: "3e-4" without a decimal point parses as a string.
        for field in ("learning_rate", "warmup_ratio", "weight_decay"):
            if field in data:
                data[field] = float(data[field])
        return cls(**data)


def load_pooled_splits() -> pd.DataFrame:
    """Concatenates samsum_v1 + dialogsum_v1 -- every row, every split; the
    caller filters to the split it needs. A `source` column is kept so
    per-dataset test evaluation (scripts/generate_m6_report.py) can still
    isolate each one."""
    frames = []
    for name in SOURCE_DATASETS:
        df = load_splits(name)
        df = df.assign(source=name)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _tokenize_split(
    df: pd.DataFrame, tokenizer: Any, max_source_length: int, max_target_length: int
) -> Dataset:
    dataset = Dataset.from_pandas(df[["dialogue", "summary"]].reset_index(drop=True))

    def tokenize_fn(batch: dict[str, list[Any]]) -> dict[str, Any]:
        prompts = [PROMPT_TEMPLATE.format(dialogue=d) for d in batch["dialogue"]]
        model_inputs = tokenizer(prompts, max_length=max_source_length, truncation=True)
        labels = tokenizer(
            text_target=batch["summary"], max_length=max_target_length, truncation=True
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return dataset.map(tokenize_fn, batched=True, remove_columns=["dialogue", "summary"])


def _make_compute_metrics(tokenizer: Any) -> Any:
    def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
        predictions = eval_pred.predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        # generate() output can carry -100 from label padding when
        # predict_with_generate reuses the label tensor shape -- clip
        # negative ids before decoding, same as the labels handling below.
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        labels = np.where(eval_pred.label_ids != -100, eval_pred.label_ids, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        metrics = compute_rouge_metrics(decoded_preds, decoded_labels)
        return {"rouge1": metrics.rouge1, "rouge2": metrics.rouge2, "rougeL": metrics.rouge_l}

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

    pooled = load_pooled_splits()
    train_df = pooled[pooled["split"] == "train"]
    val_df = pooled[pooled["split"] == "val"]

    print(
        f"model={config.model_name}: {len(train_df)} train / {len(val_df)} val "
        f"(pooled from {', '.join(SOURCE_DATASETS)}); test splits untouched here"
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    train_ds = _tokenize_split(
        train_df, tokenizer, config.max_source_length, config.max_target_length
    )
    val_ds = _tokenize_split(val_df, tokenizer, config.max_source_length, config.max_target_length)

    model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    model_slug = config.model_name.split("/")[-1]
    output_dir = MODELS_DIR / f"transformer_thread_summary_{model_slug}_v1"

    steps_per_epoch = max(len(train_df) // config.batch_size, 1)
    total_steps = config.max_steps or steps_per_epoch * config.num_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)

    training_args = Seq2SeqTrainingArguments(
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
        # Same reasoning as train_transformer.py: never resume a crashed
        # run, only the final export matters, so skip optimizer state.
        save_only_model=True,
        load_best_model_at_end=True,
        metric_for_best_model="rouge1",
        greater_is_better=True,
        predict_with_generate=True,
        generation_max_length=config.max_target_length,
        generation_num_beams=config.num_beams,
        logging_steps=50,
        report_to=[],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=_make_compute_metrics(tokenizer),
    )

    trainer.train()

    val_metrics = trainer.evaluate(val_ds, metric_key_prefix="val")
    print(
        f"final val rouge1={val_metrics['val_rouge1']:.4f} rouge2={val_metrics['val_rouge2']:.4f} rougeL={val_metrics['val_rougeL']:.4f}"
    )

    export_dir = output_dir / "final"
    trainer.save_model(str(export_dir))
    tokenizer.save_pretrained(str(export_dir))

    print(f"exported: {export_dir}")


if __name__ == "__main__":
    main()
