from collections.abc import Iterable, Iterator
from typing import TypedDict

from api.db.models import AuthorRole, TicketSource
from api.schemas.ticket import CanonicalMessage, CanonicalTicket

from ml.data.cleaning import clean_text
from ml.data.dedup import content_hash, dedup_messages
from ml.data.ids import deterministic_id
from ml.data.language import detect

DATASET_NAME = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"


class BitextRow(TypedDict):
    instruction: str
    response: str
    category: str
    intent: str
    flags: str


def _load_hf_rows() -> Iterator[BitextRow]:
    from datasets import load_dataset

    dataset = load_dataset(DATASET_NAME, split="train")
    yield from dataset


def iter_tickets(rows: Iterable[BitextRow] | None = None) -> Iterator[CanonicalTicket]:
    """rows defaults to a live Hugging Face download; tests inject a fixture instead
    so CI never downloads anything."""
    source_rows = rows if rows is not None else _load_hf_rows()

    for index, row in enumerate(source_rows):
        external_id = str(index)
        # "ticket"/"message" discriminators keep ticket ids and message ids in
        # disjoint id spaces even when their external_id strings could coincide.
        ticket_id = deterministic_id("ticket", TicketSource.BITEXT, external_id)

        instruction = clean_text(row["instruction"])
        response = clean_text(row["response"])
        instruction_lang = detect(instruction)
        response_lang = detect(response)

        customer_message = CanonicalMessage(
            id=deterministic_id("message", TicketSource.BITEXT, external_id, "0"),
            seq=0,
            author_role=AuthorRole.CUSTOMER,
            text_raw=row["instruction"],
            text_clean=instruction,
            sent_at=None,
            lang=instruction_lang.lang,
            lang_confidence=instruction_lang.confidence,
            content_hash=content_hash(instruction),
            external_id=f"{external_id}-0",
        )
        agent_message = CanonicalMessage(
            id=deterministic_id("message", TicketSource.BITEXT, external_id, "1"),
            seq=1,
            author_role=AuthorRole.AGENT,
            text_raw=row["response"],
            text_clean=response,
            sent_at=None,
            lang=response_lang.lang,
            lang_confidence=response_lang.confidence,
            content_hash=content_hash(response),
            external_id=f"{external_id}-1",
        )

        # Dedup within this ticket only, consistent with the Twitter loader —
        # a no-op in practice for Bitext (instruction and response are never
        # identical text), but keeps every loader applying the same pipeline.
        messages = list(dedup_messages([customer_message, agent_message]))
        for new_seq, message in enumerate(messages):
            message.seq = new_seq

        yield CanonicalTicket(
            id=ticket_id,
            source=TicketSource.BITEXT,
            external_id=external_id,
            created_at=None,
            channel="synthetic",
            customer_id=None,
            brand=None,
            lang=instruction_lang.lang,
            meta={
                "intent": row["intent"],
                "category": row["category"],
                "flags": row["flags"],
            },
            messages=messages,
        )
