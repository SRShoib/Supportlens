"""LLM-as-judge pass over exactly 50 real supportlens thread summaries (SPEC
M6: "an LLM-as-judge pass on exactly 50 supportlens summaries (1-5
faithfulness/coverage rubric, cached, budget ~= $0.30)"). Same shape as
ml/data/llm_seed_labels.py: gated behind LLM_ENABLED, every call goes
through the one LLMClient wrapper (persisted spend counter, hard budget
stop, cache-by-prompt-hash), never called from tests/CI (CLAUDE.md hard
rule).

Samples only from task="thread_summary" Predictions with
model_version==TRANSFORMER_MODEL_VERSION (written by
`scripts/compute_thread_summaries.py --model transformer`) -- judging the
extractive baseline is pointless, since an excerpt of the source text can't
hallucinate by construction (every word is copied verbatim); an earlier
version of this script sampled from *any* thread_summary row regardless of
which model wrote it, which silently judged 55 baseline summaries (all
trivially scoring faithfulness=5) the one time the ticket backfill hadn't
been re-run with --model transformer yet before this script ran. Filtering
by model_version here means the ticket corpus can be a mix of
baseline-summarized and transformer-summarized rows (e.g. after a --limit
transformer backfill) without this script silently drawing from the wrong
half.

Doesn't persist an EvalRun itself (same division of labor M5 established:
scripts/compute_sentiment_trajectories.py writes raw Predictions,
scripts/generate_m5_report.py aggregates and persists the eval run) --
scripts/generate_m6_report.py reads the per-ticket thread_summary_judge
Predictions this script writes and persists the aggregate.

Run (full 50-example batch): uv run python -m ml.data.llm_judge_summaries
Cheap test run first (a few cents at most):
  uv run python -m ml.data.llm_judge_summaries --limit 5
"""

import argparse
import random
import re
import sys
import uuid

from api.config import get_settings
from api.db.models import AuthorRole, Prediction, Ticket
from api.db.session import SessionLocal
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from ml.inference.base import format_dialogue
from ml.inference.llm_client import MODEL, LLMClient

PURPOSE = "thread_summary_judge"
JUDGE_SAMPLE_SIZE = 50
TRANSFORMER_MODEL_VERSION = "transformer_thread_summary_flan-t5-small_v1"
_SPEAKER_LABEL = {AuthorRole.CUSTOMER: "Customer", AuthorRole.AGENT: "Agent"}
_SYSTEM_PROMPT = (
    "You are grading an AI-generated summary of a customer support conversation on two "
    "1-5 scales. faithfulness: does the summary avoid inventing facts, numbers, or claims "
    "not present in the conversation? 5 = every claim is supported by the conversation, "
    "1 = the summary invents things that never happened. coverage: does the summary "
    "capture the conversation's key points (the customer's issue and how it was left)? "
    "5 = nothing important is missing, 1 = the summary misses the point entirely. Respond "
    "with exactly two lines, nothing else:\nfaithfulness: <1-5>\ncoverage: <1-5>"
)
_SCORE_PATTERN = re.compile(
    r"faithfulness\s*:\s*([1-5]).*?coverage\s*:\s*([1-5])", re.IGNORECASE | re.DOTALL
)


def _already_judged_ticket_ids(session: Session) -> set[uuid.UUID]:
    stmt = select(Prediction.ticket_id).where(
        Prediction.task == PURPOSE, Prediction.label == TRANSFORMER_MODEL_VERSION
    )
    return {tid for tid in session.scalars(stmt).all() if tid is not None}


def _sample_summary_predictions(session: Session, limit: int, seed: int) -> list[Prediction]:
    """Excludes tickets already judged (task=thread_summary_judge,
    label=TRANSFORMER_MODEL_VERSION) from the sampling pool -- without this,
    a small --limit dry run followed by the full run re-samples the same
    tickets deterministically (same seed, same candidate pool) and writes a
    second judge Prediction for each, since every judged ticket gets a new
    row regardless of whether the underlying LLM call was a cache hit. Found
    for real: a --limit 5 dry run before the full --limit 50 run left 5
    tickets judged twice (55 rows, 50 distinct tickets) -- see
    docs/decisions.md."""
    already_judged = _already_judged_ticket_ids(session)
    id_stmt = select(Prediction.id).where(
        Prediction.task == "thread_summary", Prediction.model_version == TRANSFORMER_MODEL_VERSION
    )
    if already_judged:
        id_stmt = id_stmt.where(Prediction.ticket_id.notin_(already_judged))
    all_ids = list(session.scalars(id_stmt).all())
    random.Random(seed).shuffle(all_ids)
    sampled_ids = all_ids[:limit]

    by_id = {
        p.id: p
        for p in session.scalars(select(Prediction).where(Prediction.id.in_(sampled_ids))).all()
    }
    return [by_id[i] for i in sampled_ids if i in by_id]


def _build_prompt(ticket: Ticket, summary: str) -> str:
    dialogue = format_dialogue(
        [(_SPEAKER_LABEL[m.author_role], m.text_clean) for m in ticket.messages]
    )
    return f"Conversation:\n{dialogue}\n\nSummary:\n{summary}"


def _parse_scores(response: str) -> tuple[int, int, bool]:
    """Returns (faithfulness, coverage, parsed_ok). Falls back to a neutral
    3/3 (not 1/1 or 5/5 -- an unparseable response is evidence of nothing,
    shouldn't read as either a pass or a hallucination) when the model
    didn't follow the requested format, same "don't crash on a malformed
    LLM response" instinct ml/data/llm_seed_labels.py's _parse_label has."""
    match = _SCORE_PATTERN.search(response)
    if match is None:
        return 3, 3, False
    return int(match.group(1)), int(match.group(2)), True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=JUDGE_SAMPLE_SIZE,
        help=(
            f"Number of summaries to judge (default {JUDGE_SAMPLE_SIZE}, ~$0.30 budget per "
            "SPEC). Lower this for a cheap dry run, e.g. --limit 5."
        ),
    )
    args = parser.parse_args()

    settings = get_settings()
    if not settings.llm_enabled:
        print(
            "LLM_ENABLED is false - skipping the paid thread-summary judge run.\n"
            "Set OPENAI_API_KEY and LLM_ENABLED=true in .env, then re-run this "
            "script when you're ready to spend the ~$0.30.",
            file=sys.stderr,
        )
        return

    session = SessionLocal()
    try:
        client = LLMClient(session, settings)
        predictions = _sample_summary_predictions(session, args.limit, settings.random_seed)
        print(
            f"judging up to {len(predictions)} not-yet-judged thread summaries "
            f"(purpose={PURPOSE}) -- already-judged tickets are skipped"
        )

        judged = 0
        cached_hits = 0
        unparsed = 0
        for prediction in predictions:
            assert prediction.ticket_id is not None  # thread_summary is always ticket-scoped
            ticket = session.scalars(
                select(Ticket)
                .options(selectinload(Ticket.messages))
                .where(Ticket.id == prediction.ticket_id)
            ).one()

            prompt = _build_prompt(ticket, prediction.label or "")
            try:
                result = client.complete(purpose=PURPOSE, prompt=prompt, system=_SYSTEM_PROMPT)
            except Exception as exc:  # BudgetExceededError, API errors, etc.
                print(f"stopping early after {judged} judged: {exc}", file=sys.stderr)
                break

            faithfulness, coverage, parsed_ok = _parse_scores(result.response)
            session.add(
                Prediction(
                    ticket_id=ticket.id,
                    task=PURPOSE,
                    label=prediction.model_version,  # which summarizer was judged
                    score=faithfulness,
                    payload={
                        "faithfulness": faithfulness,
                        "coverage": coverage,
                        "parsed_ok": parsed_ok,
                        "summary": prediction.label,
                        "cached": result.cached,
                    },
                    model_version=f"openai:{MODEL}",
                )
            )
            session.commit()
            judged += 1
            cached_hits += int(result.cached)
            unparsed += int(not parsed_ok)
            print(
                f"  ticket={ticket.id} model={prediction.model_version} "
                f"faithfulness={faithfulness} coverage={coverage}{'' if parsed_ok else ' (unparsed, fallback)'}"
            )

        print(
            f"done: {judged} judged ({cached_hits} from cache, {unparsed} unparsed), "
            f"total spend ${client.total_spend_usd():.4f}"
        )
    finally:
        session.close()


if __name__ == "__main__":
    main()
