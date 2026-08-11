"""Fine-tunes a HF token-classification model for M4 NER (SPEC M4), on
ml/data/ner/generate.py's synthetic dataset (data/splits/ner_v1.jsonl).

This is a GPU training script (CLAUDE.md ground rule #3): written and
reviewed here, but run by a human locally, never inside Docker or the API
process. See docs/m4-how-to-run-locally.md for the full setup + invocation.

Run (after `make install-training` + `make ner-data`):
  uv run python ml/training/train_token_classification.py \
      --config ml/training/configs/ner/distilbert_cased.yaml

Smoke-test only, no real training (used to sanity-check the script itself
before handing it off for a real GPU run):
  uv run python ml/training/train_token_classification.py \
      --config ml/training/configs/ner/distilbert_cased.yaml --max-steps 5
"""

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from datasets import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)

from ml.data.ner.schema import NerExample, read_jsonl
from ml.evaluation.span_metrics import compute_span_metrics
from ml.inference.rules_ner import ENTITY_LABELS
from ml.inference.token_classification import build_label_list, decode_spans

MODELS_DIR = Path("models")


@dataclass
class NerTrainConfig:
    model_name: str
    task: str = "entities"
    dataset_path: str = "data/splits/ner_v1.jsonl"
    include_paraphrases: bool = False
    paraphrase_path: str = "data/splits/ner_v1_paraphrase.jsonl"
    num_epochs: int = 4
    batch_size: int = 16
    eval_batch_size: int = 32
    learning_rate: float = 3e-5
    max_seq_length: int = 192
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_steps: int | None = None  # smoke-test override; None means "use num_epochs"

    @classmethod
    def from_yaml(cls, path: Path) -> "NerTrainConfig":
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        # PyYAML's float regex requires a decimal point, so scientific
        # notation without one (e.g. "3e-5") parses as a string -- coerce
        # explicitly (same trap ml/training/train_transformer.py works
        # around, not fixed upstream).
        for field in ("learning_rate", "warmup_ratio", "weight_decay"):
            if field in data:
                data[field] = float(data[field])
        return cls(**data)


def align_labels(
    spans: Sequence[tuple[int, int, str]],
    offsets: Sequence[Sequence[int]],
    label_to_id: dict[str, int],
) -> list[int]:
    """Projects char-span gold labels onto per-token BIO ids via the
    tokenizer's offset mapping. -100 marks special/pad tokens (offset
    width 0) so the loss ignores them. Every subword overlapping a span is
    labelled (first overlapping token gets B-, the rest get I-) rather than
    -100'ing continuation subwords -- training predicts per token and
    ml.inference.token_classification.decode_spans merges back to char
    spans via offsets, so labelling every subword makes training match the
    decode path exactly."""
    labels = [label_to_id["O"]] * len(offsets)
    for i, (start, end) in enumerate(offsets):
        if start == end:
            labels[i] = -100

    for span_start, span_end, span_label in spans:
        first = True
        for i, (start, end) in enumerate(offsets):
            if start == end:
                continue
            if start < span_end and span_start < end:
                tag = f"{'B' if first else 'I'}-{span_label}"
                labels[i] = label_to_id[tag]
                first = False
    return labels


def _build_dataset(
    examples: Sequence[NerExample], tokenizer: Any, max_length: int, label_to_id: dict[str, int]
) -> Dataset:
    raw = Dataset.from_dict(
        {
            "text": [example.text for example in examples],
            "spans": [
                [{"start": s.start, "end": s.end, "label": s.label} for s in example.entities]
                for example in examples
            ],
        }
    )

    def tokenize_fn(batch: dict[str, list[Any]]) -> dict[str, Any]:
        encoded = tokenizer(
            batch["text"], truncation=True, max_length=max_length, return_offsets_mapping=True
        )
        encoded["labels"] = [
            align_labels([(s["start"], s["end"], s["label"]) for s in spans], offsets, label_to_id)
            for spans, offsets in zip(batch["spans"], encoded["offset_mapping"], strict=True)
        ]
        return encoded

    return raw.map(tokenize_fn, batched=True, remove_columns=["text", "spans"])


@dataclass
class _EvalContext:
    """Trainer binds compute_metrics once at construction, but this script
    calls trainer.evaluate() a second time on the test split -- span
    decoding needs each example's text and offset mapping, which differ
    between the two splits, so this context is swapped between calls rather
    than baked into a fixed closure."""

    texts: list[str]
    offsets: list[list[tuple[int, int]]]


def _softmax_max(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return (exp / exp.sum(axis=-1, keepdims=True)).max(axis=-1)


def _make_compute_metrics(context: _EvalContext, labels: list[str]) -> Any:
    def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
        predicted_ids = np.argmax(eval_pred.predictions, axis=-1)
        token_scores = _softmax_max(eval_pred.predictions)

        gold_docs = []
        pred_docs = []
        for i, (text, offsets) in enumerate(zip(context.texts, context.offsets, strict=True)):
            gold_docs.append(
                decode_spans(
                    text, offsets, eval_pred.label_ids[i].tolist(), token_scores[i], labels
                )
            )
            pred_docs.append(
                decode_spans(text, offsets, predicted_ids[i].tolist(), token_scores[i], labels)
            )

        # bootstrap_resamples=0: this runs every eval epoch, only micro_f1
        # (model selection) is needed here -- the full per-entity report
        # with CIs is scripts/generate_m4_report.py's job, run once.
        metrics = compute_span_metrics(gold_docs, pred_docs, ENTITY_LABELS, bootstrap_resamples=0)
        return {"span_micro_f1": metrics.micro_f1}

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

    config = NerTrainConfig.from_yaml(args.config)
    if args.max_steps is not None:
        config.max_steps = args.max_steps

    examples = read_jsonl(Path(config.dataset_path))
    if config.include_paraphrases:
        paraphrase_path = Path(config.paraphrase_path)
        if paraphrase_path.exists():
            examples = examples + read_jsonl(paraphrase_path)
        else:
            print(f"include_paraphrases=true but {paraphrase_path} does not exist; skipping")

    train_examples = [e for e in examples if e.split == "train"]
    val_examples = [e for e in examples if e.split == "val"]
    test_examples = [e for e in examples if e.split == "test"]

    labels = build_label_list(ENTITY_LABELS)
    label_to_id = {label: i for i, label in enumerate(labels)}

    print(
        f"task={config.task} model={config.model_name}: {len(train_examples)} train / "
        f"{len(val_examples)} val / {len(test_examples)} test, {len(labels)} BIO labels"
    )

    # Deliberately not registering ml.data.masking's mask tokens as special
    # tokens (unlike scripts/compare_tokenization.py) -- an added special
    # token can yield a (0, 0) offset mid-sequence and break alignment; they
    # tokenize fine as ordinary subwords and are O-labelled regardless.
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    train_ds = _build_dataset(train_examples, tokenizer, config.max_seq_length, label_to_id)
    val_ds = _build_dataset(val_examples, tokenizer, config.max_seq_length, label_to_id)
    test_ds = _build_dataset(test_examples, tokenizer, config.max_seq_length, label_to_id)

    def offsets_of(dataset: Dataset) -> list[list[tuple[int, int]]]:
        return [[(int(s), int(e)) for s, e in row] for row in dataset["offset_mapping"]]

    eval_context = _EvalContext(texts=[e.text for e in val_examples], offsets=offsets_of(val_ds))

    # offset_mapping isn't a model input -- Trainer would pass it straight
    # into model.forward() as an unrecognized kwarg if left in.
    train_ds = train_ds.remove_columns(["offset_mapping"])
    val_ds_for_trainer = val_ds.remove_columns(["offset_mapping"])
    test_offsets = offsets_of(test_ds)
    test_ds_for_trainer = test_ds.remove_columns(["offset_mapping"])

    model = AutoModelForTokenClassification.from_pretrained(
        config.model_name,
        num_labels=len(labels),
        id2label=dict(enumerate(labels)),
        label2id=label_to_id,
    )

    model_slug = config.model_name.split("/")[-1]
    output_dir = MODELS_DIR / f"transformer_{config.task}_{model_slug}_v1"

    steps_per_epoch = max(len(train_examples) // config.batch_size, 1)
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
        # Same disk-space lesson as train_transformer.py: never resume a
        # crashed run, only ever export the final weights, so checkpoints
        # don't need optimizer/scheduler state.
        save_only_model=True,
        load_best_model_at_end=True,
        metric_for_best_model="span_micro_f1",
        greater_is_better=True,
        logging_steps=50,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds_for_trainer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=_make_compute_metrics(eval_context, labels),
    )

    trainer.train()

    eval_context.texts = [e.text for e in test_examples]
    eval_context.offsets = test_offsets
    test_metrics = trainer.evaluate(test_ds_for_trainer, metric_key_prefix="test")
    print(f"test span_micro_f1: {test_metrics['test_span_micro_f1']:.4f}")

    export_dir = output_dir / "final"
    trainer.save_model(str(export_dir))
    tokenizer.save_pretrained(str(export_dir))
    (export_dir / "label_map.json").write_text(
        json.dumps(
            {"scheme": "BIO", "labels": labels, "entity_types": sorted(ENTITY_LABELS)}, indent=2
        ),
        encoding="utf-8",
    )

    print(f"exported: {export_dir}")


if __name__ == "__main__":
    main()
