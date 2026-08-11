"""Shared example type, JSONL I/O, and the fixed-point validator every M4
data stage (synthetic generator, paraphrase pass, gold-set import) runs
examples through before trusting them.

Offline-only: imports ml.data.cleaning, which is unavailable in the API
image (infra/api.Dockerfile syncs --no-group ml). Never import this module
from apps/api or ml/inference/*.
"""

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ml.data.cleaning import clean_text
from ml.inference.rules_ner import ENTITY_LABELS


class NerValidationError(ValueError):
    """Raised by validate_example for any invariant a NER example (synthetic,
    paraphrased, or gold) must satisfy for its offsets to be trustworthy."""


@dataclass(frozen=True)
class CharSpan:
    start: int
    end: int
    label: str
    text: str


@dataclass(frozen=True)
class NerExample:
    id: str
    text: str
    entities: list[CharSpan]
    source: str
    split: str
    template_id: str | None = None


def validate_example(example: NerExample) -> None:
    """Fatal invariant checks shared by the generator, the paraphrase pass,
    and the gold-set importer. An example failing any of these can never
    produce correct offsets downstream (the offset contract documented on
    ml.inference.base.EntitySpan)."""
    text = example.text

    if clean_text(text) != text:
        raise NerValidationError(
            f"{example.id}: text is not a clean_text() fixed point -- offsets would not "
            "survive being re-cleaned at serve time"
        )

    previous_end = -1
    for span in sorted(example.entities, key=lambda s: s.start):
        if span.label not in ENTITY_LABELS:
            raise NerValidationError(f"{example.id}: unknown label {span.label!r}")
        if span.start >= span.end:
            raise NerValidationError(f"{example.id}: zero/negative-length span {span!r}")
        if span.start < 0 or span.end > len(text):
            raise NerValidationError(f"{example.id}: span {span!r} out of bounds for text")
        surface = text[span.start : span.end]
        if surface != span.text:
            raise NerValidationError(
                f"{example.id}: span {span!r} does not match text[{span.start}:{span.end}]={surface!r}"
            )
        if span.text != span.text.strip():
            raise NerValidationError(f"{example.id}: span {span!r} is whitespace-padded")
        if span.start < previous_end:
            raise NerValidationError(f"{example.id}: overlapping spans at {span!r}")
        previous_end = span.end


def _example_to_dict(example: NerExample) -> dict[str, Any]:
    return {
        "id": example.id,
        "text": example.text,
        "entities": [asdict(span) for span in example.entities],
        "source": example.source,
        "split": example.split,
        "template_id": example.template_id,
    }


def _example_from_dict(raw: dict[str, Any]) -> NerExample:
    return NerExample(
        id=raw["id"],
        text=raw["text"],
        entities=[CharSpan(**span) for span in raw["entities"]],
        source=raw["source"],
        split=raw["split"],
        template_id=raw.get("template_id"),
    )


def iter_jsonl(path: Path) -> Iterator[NerExample]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                yield _example_from_dict(json.loads(stripped))


def read_jsonl(path: Path) -> list[NerExample]:
    return list(iter_jsonl(path))


def write_jsonl(examples: Iterable[NerExample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(_example_to_dict(example)) + "\n")


def append_jsonl(examples: Iterable[NerExample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(_example_to_dict(example)) + "\n")
