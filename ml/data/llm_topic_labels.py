"""Optional LLM pass that turns a topic's c-TF-IDF keyword list into a
short human-readable name (SPEC M7: "optional one capped LLM pass to name
the top 30 topics", SPEC §5 budget ~= $0.20). Every call goes through the
one LLMClient wrapper (persisted spend counter, hard budget stop,
cache-by-prompt-hash), gated behind LLM_ENABLED, never called from
tests/CI (CLAUDE.md hard rule).

Default is OFF: the c-TF-IDF keyword-joined label
(ml/training/topic_model.py's _label_from_keywords) is what
scripts/assign_topics.py writes and what the dashboard shows out of the
box. Running this script is a deliberate, later, budget-spending
enhancement -- it UPDATEs Topic.label in place for the current top-N
topics by size, leaving `keywords` (the underlying c-TF-IDF terms)
untouched.

Unlike ml/data/llm_judge_summaries.py, this script does NOT need an
"already labeled" exclusion guard (see docs/decisions.md's postmortem on
that script's duplicate-insert bug). That bug was specific to *random
sampling* into a shrinking candidate pool across separate INSERT-based
runs -- a small dry run and a later full run could draw overlapping
tickets and each write its own new Prediction row. This script always
processes the same deterministic top-N-by-size topics and UPDATEs
Topic.label in place rather than inserting a new row: a re-run overwrites
the same row with the same (cache-hit, free) result, which is naturally
idempotent without any extra bookkeeping.

Run (top 30 topics, ~$0.20 budget per SPEC):
  uv run python -m ml.data.llm_topic_labels
Cheap test run first:
  uv run python -m ml.data.llm_topic_labels --limit 3
"""

import argparse
import sys

from api.config import get_settings
from api.db.models import Topic
from api.db.session import SessionLocal
from sqlalchemy import select
from sqlalchemy.orm import Session

from ml.inference.llm_client import LLMClient

PURPOSE = "topic_llm_label"
DEFAULT_TOP_N = 30
OUTLIER_TOPIC_KEY = -1
_SYSTEM_PROMPT = (
    "You are naming a topic cluster from a customer support ticket corpus. Given a ranked "
    "list of keywords that characterize the cluster, respond with a short human-readable "
    "topic name (2-5 words, title case, no ending punctuation, no quotes). Respond with the "
    "name only, nothing else."
)


def _top_topics(session: Session, limit: int) -> list[Topic]:
    stmt = (
        select(Topic)
        .where(Topic.topic_key != OUTLIER_TOPIC_KEY)
        .order_by(Topic.size.desc())
        .limit(limit)
    )
    return list(session.scalars(stmt).all())


def _build_prompt(topic: Topic) -> str:
    return f"Keywords (most to least characteristic): {', '.join(topic.keywords)}"


def _clean_name(response: str) -> str:
    """Strips wrapping quotes/whitespace/trailing punctuation an LLM
    sometimes adds despite the system prompt asking for none -- mirrors
    ml/data/llm_seed_labels.py's "don't trust the model to follow
    formatting instructions exactly" instinct."""
    return response.strip().strip("\"'").strip(".").strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_TOP_N,
        help=(
            f"Number of topics to name, ranked by size (default {DEFAULT_TOP_N}, ~$0.20 "
            "budget per SPEC). Lower this for a cheap dry run, e.g. --limit 3."
        ),
    )
    args = parser.parse_args()

    settings = get_settings()
    if not settings.llm_enabled:
        print(
            "LLM_ENABLED is false - skipping the paid topic-naming pass.\n"
            "Set OPENAI_API_KEY and LLM_ENABLED=true in .env, then re-run this "
            "script when you're ready to spend the ~$0.20.",
            file=sys.stderr,
        )
        return

    session = SessionLocal()
    try:
        client = LLMClient(session, settings)
        topics = _top_topics(session, args.limit)
        print(f"naming {len(topics)} topics (purpose={PURPOSE})")

        named = 0
        cached_hits = 0
        for topic in topics:
            try:
                result = client.complete(
                    purpose=PURPOSE, prompt=_build_prompt(topic), system=_SYSTEM_PROMPT
                )
            except Exception as exc:  # BudgetExceededError, API errors, etc.
                print(f"stopping early after {named} named: {exc}", file=sys.stderr)
                break

            cleaned = _clean_name(result.response)
            old_label = topic.label
            if cleaned:
                topic.label = cleaned
                session.commit()
            named += 1
            cached_hits += int(result.cached)
            print(f"  topic_key={topic.topic_key}: {old_label!r} -> {topic.label!r}")

        print(
            f"done: {named} named ({cached_hits} from cache), "
            f"total spend ${client.total_spend_usd():.4f}"
        )
    finally:
        session.close()


if __name__ == "__main__":
    main()
