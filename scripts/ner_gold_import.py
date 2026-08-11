"""Imports the hand-annotated data/gold/ner_gold_v1.todo.md (produced by
scripts/ner_gold_export.py) into the committed gold set,
data/gold/ner_gold_v1.jsonl (SPEC M4; CLAUDE.md explicitly permits
committing this one dataset, unlike everything else under data/). Validates
before writing anything -- a failed import or validation writes nothing, so
a bad hand-edit can never silently corrupt the gold set.

Run (after hand-annotating data/gold/ner_gold_v1.todo.md):
  uv run python scripts/ner_gold_import.py
"""

import json
import sys
from pathlib import Path

from ml.data.ner.gold import GoldImportError, GoldSetError, import_gold_markdown, validate_gold_set
from ml.data.ner.schema import NerValidationError, read_jsonl, write_jsonl

MARKDOWN_PATH = Path("data/gold/ner_gold_v1.todo.md")
MANIFEST_PATH = Path("data/gold/ner_gold_v1.meta.json")
OUTPUT_PATH = Path("data/gold/ner_gold_v1.jsonl")
SYNTHETIC_PATH = Path("data/splits/ner_v1.jsonl")


def main() -> None:
    if not MARKDOWN_PATH.exists() or not MANIFEST_PATH.exists():
        print(
            f"{MARKDOWN_PATH} and {MANIFEST_PATH} must both exist -- "
            "run scripts/ner_gold_export.py first.",
            file=sys.stderr,
        )
        return

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    markdown = MARKDOWN_PATH.read_text(encoding="utf-8")

    try:
        examples = import_gold_markdown(markdown, manifest["originals"])
    except GoldImportError as exc:
        print(f"import failed: {exc}", file=sys.stderr)
        return

    synthetic_ids = (
        [example.id for example in read_jsonl(SYNTHETIC_PATH)] if SYNTHETIC_PATH.exists() else []
    )
    if not synthetic_ids:
        print(
            f"warning: {SYNTHETIC_PATH} not found -- skipping the leakage check against the "
            "synthetic training set",
            file=sys.stderr,
        )

    try:
        report = validate_gold_set(examples, synthetic_ids=synthetic_ids)
    except (GoldSetError, NerValidationError) as exc:
        print(f"validation failed: {exc}", file=sys.stderr)
        return

    write_jsonl(examples, OUTPUT_PATH)
    print(f"wrote {report.n_examples} examples -> {OUTPUT_PATH}")
    print(f"span counts: {report.span_counts_by_label}")
    for warning in report.warnings:
        print(f"warning: {warning}", file=sys.stderr)


if __name__ == "__main__":
    main()
