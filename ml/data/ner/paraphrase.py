"""Optional budget-gated LLM paraphrase pass over ml/data/ner/generate.py's
templated examples (SPEC M4 section 2; SPEC section 5 budgets ~$0.50 for
this -- actual cost at gpt-4o-mini rates is closer to $0.06 per 1,000
calls). Rewrites a template's often-robotic phrasing into a more natural
customer-support voice while keeping every entity's surface form and label
byte-identical, so gold offsets are recomputed from the parsed response
rather than trusted from the model's own claims.

Strictly optional: ml/data/ner/generate.py's output alone is a complete,
trainable dataset. This produces a second, additive file -- training
defaults to NOT including it (ml/training/configs/ner/*.yaml's
include_paraphrases: false) until a human explicitly opts in.

Gated behind LLM_ENABLED -- refuses to spend money until you opt in by
setting OPENAI_API_KEY and LLM_ENABLED=true in .env.

Run (full batch, ~$0.06/1000 calls at gpt-4o-mini pricing):
  uv run python -m ml.data.ner.paraphrase
Cheap test run first:
  uv run python -m ml.data.ner.paraphrase --limit 20
"""

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from api.config import get_settings
from api.db.session import SessionLocal
from sqlalchemy.orm import Session

from ml.data.ner.markup import MarkupError, parse, render
from ml.data.ner.schema import (
    NerExample,
    NerValidationError,
    append_jsonl,
    iter_jsonl,
    read_jsonl,
    validate_example,
)
from ml.inference.llm_client import LLMClient

PURPOSE = "ner_paraphrase"
INPUT_PATH = Path("data/splits/ner_v1.jsonl")
OUTPUT_PATH = Path("data/splits/ner_v1_paraphrase.jsonl")
DEFAULT_LIMIT = 1000

_SYSTEM_PROMPT = (
    "You rewrite short customer-support sentences into a more natural, casual voice. "
    "The input contains bracketed entity annotations in the form [surface|LABEL]. "
    "You MUST reproduce every bracketed block byte-for-byte exactly once, in any order "
    "that reads naturally -- do not change the text inside the brackets, do not add new "
    "bracketed blocks, do not drop any. Only rewrite the plain text around them. Respond "
    "with only the rewritten sentence, nothing else."
)


@dataclass(frozen=True)
class PassStats:
    accepted: int
    rejected: int
    cached_hits: int
    total_spend_usd: float


def _candidate_examples(path: Path) -> list[NerExample]:
    """Only template-sourced examples are paraphrased -- shell-injected ones
    already carry real, naturally messy surrounding text."""
    return [example for example in read_jsonl(path) if example.source == "template"]


def _paraphrase_id(source_id: str) -> str:
    return f"para:{source_id}"


def _already_paraphrased_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {example.id for example in iter_jsonl(path)}


def _try_paraphrase(example: NerExample, response: str) -> NerExample | None:
    """Re-parses the model's response and validates it independently of the
    model's own claims about what it did. Returns None (discard, never
    repair) on a parse failure, a dropped/duplicated/altered entity block,
    or any offset-contract violation."""
    try:
        text, spans = parse(response)
    except MarkupError:
        return None

    if sorted((s.label, s.text) for s in example.entities) != sorted(
        (s.label, s.text) for s in spans
    ):
        return None

    candidate = NerExample(
        id=_paraphrase_id(example.id),
        text=text,
        entities=spans,
        source="paraphrase",
        # Inherits the parent's split -- a paraphrase crossing into a
        # different split than its own source example would be leakage.
        split=example.split,
        template_id=example.template_id,
    )
    try:
        validate_example(candidate)
    except NerValidationError:
        return None
    return candidate


def run_paraphrase_pass(
    client: LLMClient, candidates: Sequence[NerExample], output_path: Path
) -> PassStats:
    accepted = 0
    rejected = 0
    cached_hits = 0

    for example in candidates:
        markup = render(example.text, example.entities)
        try:
            result = client.complete(purpose=PURPOSE, prompt=markup, system=_SYSTEM_PROMPT)
        except Exception as exc:  # BudgetExceededError, API errors, etc.
            print(f"stopping early after {accepted + rejected} attempts: {exc}", file=sys.stderr)
            break

        cached_hits += int(result.cached)
        candidate = _try_paraphrase(example, result.response)
        if candidate is None:
            rejected += 1
            continue

        append_jsonl([candidate], output_path)
        accepted += 1

    return PassStats(
        accepted=accepted,
        rejected=rejected,
        cached_hits=cached_hits,
        total_spend_usd=client.total_spend_usd(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=(
            f"Number of template examples to paraphrase (default {DEFAULT_LIMIT}). Lower this "
            "for a cheap dry run, e.g. --limit 20 -- costs a fraction of a cent and pre-fills "
            "the cache, so a later full run reuses those paraphrases instead of re-billing."
        ),
    )
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    settings = get_settings()
    if not settings.llm_enabled:
        print(
            "LLM_ENABLED is false - skipping the paid NER paraphrase pass.\n"
            "Set OPENAI_API_KEY and LLM_ENABLED=true in .env, then re-run this "
            "script when you're ready to spend a few cents.",
            file=sys.stderr,
        )
        return

    if not args.input.exists():
        print(f"{args.input} does not exist -- run `make ner-data` first.", file=sys.stderr)
        return

    already_done = _already_paraphrased_ids(args.output)
    candidates = [
        example
        for example in _candidate_examples(args.input)
        if _paraphrase_id(example.id) not in already_done
    ][: args.limit]
    print(f"paraphrasing up to {len(candidates)} template examples (purpose={PURPOSE})")

    session: Session = SessionLocal()
    try:
        client = LLMClient(session, settings)
        stats = run_paraphrase_pass(client, candidates, args.output)
    finally:
        session.close()

    print(
        f"done: {stats.accepted} accepted, {stats.rejected} rejected "
        f"({stats.cached_hits} from cache), total spend ${stats.total_spend_usd:.4f}"
    )


if __name__ == "__main__":
    main()
