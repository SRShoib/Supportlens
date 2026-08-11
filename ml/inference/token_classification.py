"""HF token-classification inference wrapper for M4 NER (SPEC M4: "Fine-tune
a HF token-classification model locally... serve on CPU"). Mirrors
ml/inference/transformer.py's shape: loads an exported model + tokenizer
once, serves predict() on CPU.

Never cleans its input: start/end are always character offsets into the
exact string passed to predict() -- the offset contract every M4 component
shares (see ml/inference/base.py's EntitySpan docstring; ml/inference/rules_ner.py's
module docstring documents the other implementation of the same contract).
"""

import json
from collections.abc import Sequence
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from ml.inference.base import EntityResult, EntitySpan


def build_label_list(entity_types: Sequence[str]) -> list[str]:
    """BIO labels, "O" first, then B-/I- per type in a fixed (alphabetical)
    order -- an explicit, independently-checkable label order, not left to
    dict/set iteration order. The canonical definition of the label scheme
    decode_spans() consumes: ml/training/train_token_classification.py uses
    this to build training labels, and scripts/make_stub_models.py uses it
    to build the stub_ner fixture's label_map.json, so the scheme is
    defined in exactly one place rather than duplicated (and potentially
    drifting) across the training script and the fixture generator."""
    labels = ["O"]
    for entity_type in sorted(entity_types):
        labels.append(f"B-{entity_type}")
        labels.append(f"I-{entity_type}")
    return labels


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    surface = text[start:end]
    stripped = surface.strip()
    if not stripped:
        return start, start
    left_pad = surface.index(stripped)
    return start + left_pad, start + left_pad + len(stripped)


def decode_spans(
    text: str,
    offsets: Sequence[tuple[int, int]],
    label_ids: Sequence[int],
    token_scores: Sequence[float],
    labels: Sequence[str],
) -> list[EntitySpan]:
    """Merges a per-token BIO label sequence into char spans. Kept a pure
    function, separate from the model class, because that's where the real
    test coverage lives -- a stub checkpoint's random weights predict
    meaningless labels, so the decoder's correctness is what actually
    matters, and this same function is what ml/training/train_token_classification.py's
    compute_metrics uses to score validation during training. That sharing
    is deliberate: model selection, offline evaluation, and serving all
    merge tokens into spans exactly the same way.

    IOB2 with permissive repair: a bare I-X with nothing of type X currently
    open still opens a span, rather than being dropped -- a model is fully
    capable of predicting a well-formed sequence that just never emits the
    B- tag first, and penalizing that at decode time would be double
    -counting a labeling-scheme technicality as a real error. `score` is the
    mean per-token max-softmax probability across the span. Every returned
    span satisfies text[start:end] == span.text by construction."""
    spans: list[EntitySpan] = []
    open_start: int | None = None
    open_end: int | None = None
    open_type: str | None = None
    open_scores: list[float] = []

    def close() -> None:
        nonlocal open_start, open_end, open_type, open_scores
        if open_type is not None and open_start is not None and open_end is not None:
            start, end = _trim(text, open_start, open_end)
            if start < end:
                spans.append(
                    EntitySpan(
                        start=start,
                        end=end,
                        label=open_type,
                        text=text[start:end],
                        score=sum(open_scores) / len(open_scores),
                    )
                )
        open_start = open_end = open_type = None
        open_scores = []

    for i, (start, end) in enumerate(offsets):
        if start == end:  # special/pad token -- never part of a span
            close()
            continue
        label_id = label_ids[i]
        if label_id < 0 or label_id >= len(labels):
            close()
            continue
        tag = labels[label_id]
        if tag == "O":
            close()
            continue

        prefix, _, entity_type = tag.partition("-")
        if prefix == "B" or entity_type != open_type:
            close()
            open_start, open_end, open_type = start, end, entity_type
            open_scores = [token_scores[i]]
        else:
            open_end = end
            open_scores.append(token_scores[i])

    close()
    return spans


class TokenClassificationPredictor:
    """Loads an exported HF token-classification model + tokenizer once and
    serves predict() on CPU (SPEC §3: serving is CPU-only by design). Reads
    label_map.json (written by ml/training/train_token_classification.py)
    for the BIO label order -- the same pattern TransformerPredictor uses
    for classification label order, an explicit, independently-checkable
    artifact rather than an HF-internal detail. Does NOT register the
    <URL>/<USER>/<EMAIL>/<PHONE> mask tokens as special tokens the way
    scripts/compare_tokenization.py does: an added special token can produce
    a (0, 0) offset and break the alignment decode_spans relies on, and
    they're all O-labelled anyway."""

    def __init__(self, export_dir: Path, max_length: int = 192) -> None:
        if not (export_dir / "model.safetensors").exists():
            raise FileNotFoundError(f"no exported model at {export_dir}")

        self._tokenizer = AutoTokenizer.from_pretrained(str(export_dir))
        self._model = AutoModelForTokenClassification.from_pretrained(str(export_dir))
        self._model.eval()
        self._max_length = max_length
        label_map = json.loads((export_dir / "label_map.json").read_text(encoding="utf-8"))
        self._labels: list[str] = label_map["labels"]

    def predict(self, texts: list[str]) -> list[EntityResult]:
        if not texts:
            return []

        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self._max_length,
            return_offsets_mapping=True,
        )
        offsets_mapping = inputs.pop("offset_mapping")

        with torch.no_grad():
            logits = self._model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)
        predicted_ids = probabilities.argmax(dim=-1)
        token_scores = probabilities.max(dim=-1).values

        results = []
        for i, text in enumerate(texts):
            offsets = [(int(pair[0]), int(pair[1])) for pair in offsets_mapping[i].tolist()]
            entities = decode_spans(
                text, offsets, predicted_ids[i].tolist(), token_scores[i].tolist(), self._labels
            )
            results.append(
                EntityResult(entities=entities, truncated=self._is_truncated(text, offsets))
            )
        return results

    @staticmethod
    def _is_truncated(text: str, offsets: Sequence[tuple[int, int]]) -> bool:
        real_offsets = [o for o in offsets if o[0] != o[1]]
        if not real_offsets:
            return False
        return real_offsets[-1][1] < len(text)
