"""Seed-labels urgency via a small LLM sample (SPEC §2: <=2,000 examples, ~$0.50).

Gated behind LLM_ENABLED — refuses to spend money until you opt in by setting
OPENAI_API_KEY and LLM_ENABLED=true in .env.

Run: uv run python -m ml.data.llm_seed_labels
"""

import random
import sys

from api.config import get_settings
from api.db.models import AuthorRole, Message, Prediction, Ticket, TicketSource
from api.db.session import SessionLocal
from sqlalchemy import select
from sqlalchemy.orm import Session

from ml.inference.llm_client import MODEL, LLMClient

PURPOSE = "urgency_llm_seed"
SEED_SET_SIZE = 2000
_VALID_LABELS = {"low", "medium", "high"}
_SYSTEM_PROMPT = (
    "You are labeling the urgency of a customer support message. Respond with "
    "exactly one word: low, medium, or high. low = routine question, no "
    "frustration. medium = annoyed, wants resolution soon. high = threatening "
    "legal/refund action, fraud/security concern, ALL-CAPS anger, or repeated "
    "unresolved contact."
)


def _sample_messages(session: Session, limit: int, seed: int) -> list[Message]:
    id_stmt = (
        select(Message.id)
        .join(Ticket, Ticket.id == Message.ticket_id)
        .where(Ticket.source == TicketSource.TWITTER, Message.author_role == AuthorRole.CUSTOMER)
    )
    all_ids = list(session.scalars(id_stmt).all())
    random.Random(seed).shuffle(all_ids)
    sampled_ids = all_ids[:limit]

    by_id = {
        m.id: m for m in session.scalars(select(Message).where(Message.id.in_(sampled_ids))).all()
    }
    return [by_id[i] for i in sampled_ids if i in by_id]


def _parse_label(response: str) -> str:
    label = response.strip().lower()
    return label if label in _VALID_LABELS else "medium"


def main() -> None:
    settings = get_settings()
    if not settings.llm_enabled:
        print(
            "LLM_ENABLED is false - skipping the paid urgency seed-labeling run.\n"
            "Set OPENAI_API_KEY and LLM_ENABLED=true in .env, then re-run this "
            "script when you're ready to spend the ~$0.50.",
            file=sys.stderr,
        )
        return

    session = SessionLocal()
    try:
        client = LLMClient(session, settings)
        messages = _sample_messages(session, SEED_SET_SIZE, settings.random_seed)
        print(f"seed-labeling up to {len(messages)} messages (purpose={PURPOSE})")

        labeled = 0
        cached_hits = 0
        for message in messages:
            try:
                result = client.complete(
                    purpose=PURPOSE, prompt=message.text_clean, system=_SYSTEM_PROMPT
                )
            except Exception as exc:  # BudgetExceededError, API errors, etc.
                print(f"stopping early after {labeled} labels: {exc}", file=sys.stderr)
                break

            label = _parse_label(result.response)
            session.add(
                Prediction(
                    message_id=message.id,
                    task=PURPOSE,
                    label=label,
                    model_version=f"openai:{MODEL}",
                    payload={"cached": result.cached},
                )
            )
            session.commit()
            labeled += 1
            cached_hits += int(result.cached)

        print(
            f"done: {labeled} labeled ({cached_hits} from cache), "
            f"total spend ${client.total_spend_usd():.4f}"
        )
    finally:
        session.close()


if __name__ == "__main__":
    main()
