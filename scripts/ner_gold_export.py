"""Exports gold-set annotation candidates as a human-editable Markdown file
(SPEC M4: "hand-verify a 200-example gold test set"). Draws message ids from
the held-out gold pool ml/data/ner/generate.py reserved
(data/splits/ner_pools_v1.json) -- disjoint from every message used in
training by construction, not by convention.

Run (needs `make ner-data` to have run first, so the partition file exists):
  uv run python scripts/ner_gold_export.py

Then hand-annotate data/gold/ner_gold_v1.todo.md directly per
docs/ner-annotation-guidelines.md, and run scripts/ner_gold_import.py.
"""

import json
import random
import sys
from pathlib import Path

from api.config import get_settings
from api.db.models import Message
from api.db.session import SessionLocal
from sqlalchemy import select
from sqlalchemy.orm import Session

from ml.data.ner.gold import (
    GUIDELINE_VERSION,
    GoldCandidate,
    propose_entities,
    render_gold_markdown,
    select_gold_candidates,
)

PARTITION_PATH = Path("data/splits/ner_pools_v1.json")
MARKDOWN_PATH = Path("data/gold/ner_gold_v1.todo.md")
MANIFEST_PATH = Path("data/gold/ner_gold_v1.meta.json")


def _load_gold_ids(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data["gold_ids"])


def _fetch_texts(session: Session, message_ids: list[str]) -> dict[str, str]:
    stmt = select(Message.id, Message.text_clean).where(Message.id.in_(message_ids))
    return {str(message_id): text for message_id, text in session.execute(stmt).all()}


def _load_spacy() -> object | None:
    """Optional: falls back to rules-only proposals if spaCy or its model
    isn't installed rather than failing the whole export -- annotation can
    still proceed, just with one fewer proposer's suggestions."""
    try:
        import spacy
    except ImportError:
        print(
            "spacy not installed - proposing entities from the rules baseline only",
            file=sys.stderr,
        )
        return None
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        print(
            "en_core_web_sm not installed - proposing entities from the rules baseline only",
            file=sys.stderr,
        )
        return None


def main() -> None:
    if not PARTITION_PATH.exists():
        print(f"{PARTITION_PATH} does not exist -- run `make ner-data` first.", file=sys.stderr)
        return

    settings = get_settings()
    gold_ids = _load_gold_ids(PARTITION_PATH)

    session = SessionLocal()
    try:
        texts = _fetch_texts(session, gold_ids)
    finally:
        session.close()

    nlp = _load_spacy()
    candidates = [
        GoldCandidate(id=message_id, text=text, proposed_entities=propose_entities(text, nlp))
        for message_id, text in texts.items()
    ]

    rng = random.Random(settings.random_seed)
    selection = select_gold_candidates(candidates, rng)

    span_counts: dict[str, int] = {}
    for candidate in selection.candidates:
        for span in candidate.proposed_entities:
            span_counts[span.label] = span_counts.get(span.label, 0) + 1

    MARKDOWN_PATH.parent.mkdir(parents=True, exist_ok=True)
    MARKDOWN_PATH.write_text(render_gold_markdown(selection), encoding="utf-8")
    MANIFEST_PATH.write_text(
        json.dumps(
            {
                "guideline_version": GUIDELINE_VERSION,
                "seed": settings.random_seed,
                "candidate_pool_size": len(candidates),
                "selected_count": len(selection.candidates),
                "blind_ids": sorted(selection.blind_ids),
                "proposed_span_counts": span_counts,
                "originals": {c.id: c.text for c in selection.candidates},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"wrote {len(selection.candidates)} candidates -> {MARKDOWN_PATH}")
    print(f"blind subset: {len(selection.blind_ids)} ids (no suggestions shown)")
    print(f"proposed span counts: {span_counts}")
    print(f"manifest -> {MANIFEST_PATH}")
    print(
        f"next: hand-annotate {MARKDOWN_PATH} per docs/ner-annotation-guidelines.md, "
        "then run scripts/ner_gold_import.py"
    )


if __name__ == "__main__":
    main()
