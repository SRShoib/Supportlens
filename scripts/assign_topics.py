"""Writes a fitted M7 topic model's output into Postgres: the topic catalog
(labels + c-TF-IDF keywords, SPEC M7's "human-readable topic labels") into
`topics`, and one Prediction(task="topic") per ticket -- matching M5/M6's
convention of a durably-stored per-ticket Prediction the dashboard reads,
never recomputed live (apps/api never loads an embedding or topic model,
see docs/decisions.md).

Reads exactly the pair ml/training/topic_model.py's export_variant()
writes -- models/topics_{variant}_v1/{topics.json,assignments.parquet} --
so this script does no ML itself (no embedding, no clustering) and needs
none of the `topics` dependency group's heavy libraries, only pandas
(already a default dep). "Assignment" here means "read back the assignment
already computed at fit time", not "classify a new ticket" -- SPEC M7 fits
and assigns the whole corpus in one pass, there's no held-out inference
step.

Full-recompute semantics: every run deletes ALL existing task="topic"
Predictions and ALL existing `topics` rows before reinserting -- same
rationale as scripts/compute_sentiment_trajectories.py (Prediction/Topic
ids are random UUIDs, re-running isn't naturally a no-op). This also means
only one variant's assignment is ever live in Postgres at a time; running
`--variant kmeans` then `--variant bertopic` fully replaces the former with
the latter, it doesn't leave both.

No --limit flag here (unlike the other M7 scripts): unlike
compute_embeddings.py, there's no per-ticket model inference happening in
this script -- it's a bulk read of two already-computed flat files and a
bulk insert, fast at full corpus size, so there's nothing expensive to cap.

Run:
  uv run python scripts/assign_topics.py                  # --variant bertopic (default)
  uv run python scripts/assign_topics.py --variant kmeans
"""

import argparse
import json
import uuid
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from api.db.models import Prediction, Topic
from api.db.session import SessionLocal
from sqlalchemy import delete
from sqlalchemy.orm import Session

MODELS_DIR = Path("models")
TASK = "topic"
VARIANTS = ("kmeans", "bertopic")
DEFAULT_VARIANT = "bertopic"  # SPEC M7's named pipeline; "kmeans" is the CLAUDE.md-mandated
# comparison baseline only, never the intended deployment (docs/decisions.md).


@dataclass(frozen=True)
class TopicCatalogEntry:
    topic_key: int
    label: str
    keywords: list[str]
    size: int


@dataclass(frozen=True)
class TopicAssignment:
    ticket_id: str
    topic_key: int
    probability: float


def load_topic_catalog(variant_dir: Path) -> tuple[str, list[TopicCatalogEntry]]:
    raw = json.loads((variant_dir / "topics.json").read_text(encoding="utf-8"))
    entries = [
        TopicCatalogEntry(t["topic_key"], t["label"], t["keywords"], t["size"])
        for t in raw["topics"]
    ]
    return raw["model_version"], entries


def load_assignments(variant_dir: Path) -> list[TopicAssignment]:
    df = pd.read_parquet(variant_dir / "assignments.parquet")
    return [
        TopicAssignment(str(row.ticket_id), int(row.topic_key), float(row.probability))
        for row in df.itertuples()
    ]


def persist_topic_assignments(
    session: Session,
    model_version: str,
    catalog: list[TopicCatalogEntry],
    assignments: list[TopicAssignment],
) -> int:
    session.execute(delete(Prediction).where(Prediction.task == TASK))
    session.execute(delete(Topic))

    for entry in catalog:
        session.add(
            Topic(
                topic_key=entry.topic_key,
                label=entry.label,
                keywords=entry.keywords,
                size=entry.size,
                model_version=model_version,
            )
        )

    by_key = {entry.topic_key: entry for entry in catalog}
    written = 0
    for assignment in assignments:
        entry = by_key.get(assignment.topic_key)
        session.add(
            Prediction(
                ticket_id=uuid.UUID(assignment.ticket_id),
                task=TASK,
                label=entry.label if entry else "unlabeled",
                score=assignment.probability,
                payload={
                    "topic_key": assignment.topic_key,
                    "keywords": entry.keywords if entry else [],
                },
                model_version=model_version,
            )
        )
        written += 1

    session.commit()
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, default=DEFAULT_VARIANT)
    args = parser.parse_args()

    variant_dir = MODELS_DIR / f"topics_{args.variant}_v1"
    model_version, catalog = load_topic_catalog(variant_dir)
    assignments = load_assignments(variant_dir)
    n_topics = len({e.topic_key for e in catalog if e.topic_key != -1})
    print(f"loaded {n_topics} topics + {len(assignments)} ticket assignments from {variant_dir}")

    session = SessionLocal()
    try:
        written = persist_topic_assignments(session, model_version, catalog, assignments)
        print(f"wrote {len(catalog)} topics and {written} {TASK} predictions ({model_version})")
    finally:
        session.close()


if __name__ == "__main__":
    main()
