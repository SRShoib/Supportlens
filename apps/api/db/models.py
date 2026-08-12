import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy import Enum as SAEnum
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from api.db.base import Base


class TicketSource(StrEnum):
    BITEXT = "bitext"
    TWITTER = "twitter"


class AuthorRole(StrEnum):
    CUSTOMER = "customer"
    AGENT = "agent"


def _enum(enum_cls: type[StrEnum], name: str) -> SAEnum:
    return SAEnum(enum_cls, name=name, values_callable=lambda cls: [e.value for e in cls])


class Ticket(Base):
    __tablename__ = "tickets"
    __table_args__ = (
        UniqueConstraint("source", "external_id", name="tickets_source_external_id"),
        Index("ix_tickets_created_at", "created_at"),
        Index("ix_tickets_brand", "brand"),
    )

    id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True)
    source: Mapped[TicketSource] = mapped_column(
        _enum(TicketSource, "ticket_source"), nullable=False, index=True
    )
    external_id: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    channel: Mapped[str] = mapped_column(String, nullable=False)
    customer_id: Mapped[str | None] = mapped_column(String, nullable=True)
    brand: Mapped[str | None] = mapped_column(String, nullable=True)
    lang: Mapped[str | None] = mapped_column(String(8), nullable=True)
    meta: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    messages: Mapped[list["Message"]] = relationship(
        back_populates="ticket", cascade="all, delete-orphan", order_by="Message.seq"
    )


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (
        UniqueConstraint("ticket_id", "seq", name="messages_ticket_id_seq"),
        Index("ix_messages_content_hash", "content_hash"),
        Index("ix_messages_lang", "lang"),
    )

    id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True)
    ticket_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("tickets.id", ondelete="CASCADE"), nullable=False
    )
    seq: Mapped[int] = mapped_column(nullable=False)
    author_role: Mapped[AuthorRole] = mapped_column(
        _enum(AuthorRole, "author_role"), nullable=False
    )
    text_raw: Mapped[str] = mapped_column(String, nullable=False)
    text_clean: Mapped[str] = mapped_column(String, nullable=False)
    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    lang: Mapped[str | None] = mapped_column(String(8), nullable=True)
    lang_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    content_hash: Mapped[str] = mapped_column(String(40), nullable=False)
    external_id: Mapped[str] = mapped_column(String, nullable=False)
    meta: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)

    ticket: Mapped[Ticket] = relationship(back_populates="messages")


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (
        CheckConstraint(
            "(ticket_id IS NOT NULL) OR (message_id IS NOT NULL)", name="prediction_has_target"
        ),
        Index("ix_predictions_ticket_task", "ticket_id", "task"),
        Index("ix_predictions_task_model_version", "task", "model_version"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    ticket_id: Mapped[uuid.UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("tickets.id", ondelete="CASCADE"), nullable=True
    )
    message_id: Mapped[uuid.UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("messages.id", ondelete="CASCADE"), nullable=True
    )
    task: Mapped[str] = mapped_column(String, nullable=False)
    label: Mapped[str | None] = mapped_column(String, nullable=True)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    model_version: Mapped[str] = mapped_column(String, nullable=False)
    eval_run_id: Mapped[uuid.UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("eval_runs.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Topic(Base):
    """One topic from a fitted M7 topic model (BERTopic or the TF-IDF/
    KMeans baseline) -- label + c-TF-IDF keywords, written once per topic
    by scripts/assign_topics.py, read by GET /topics. Per-ticket topic
    assignment is a Prediction(task="topic") row instead (matching M5/M6's
    convention), not a column here -- this table is the small topic
    catalog, not a per-ticket fact table. `topic_key` mirrors BERTopic's
    own topic ids, where -1 is the HDBSCAN outlier cluster (never counted
    toward SPEC M7's "≥ 30 coherent topics")."""

    __tablename__ = "topics"
    __table_args__ = (
        UniqueConstraint("model_version", "topic_key", name="topics_model_version_topic_key"),
        Index("ix_topics_model_version", "model_version"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    topic_key: Mapped[int] = mapped_column(nullable=False)
    label: Mapped[str] = mapped_column(String, nullable=False)
    keywords: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    size: Mapped[int] = mapped_column(nullable=False)
    model_version: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class KbArticle(Base):
    """A synthetic knowledge-base article (SPEC M8: "a small synthetic KB
    (~40 articles)") -- generated once by ml/data/kb_generate.py, purely
    templated (no LLM cost; SPEC §5's M8 budget line is earmarked for RAG
    reply drafting, not KB writing, see docs/decisions.md). Postgres is the
    canonical source, same "canonical table, Chroma is a derived index"
    relationship Ticket/Message already have with M7's embeddings --
    scripts/index_search_corpus.py reads these rows and pushes them into the
    `kb_articles` Chroma collection, it never writes here.

    `id` is deterministic (ml/data/ids.py::deterministic_id, keyed off
    generator_version + source_kind + source_key), not random uuid4 like
    Topic/Prediction -- kb_generate.py is fully reproducible from its inputs,
    so re-running it is a natural upsert instead of needing a
    delete-everything-first step."""

    __tablename__ = "kb_articles"
    __table_args__ = (
        UniqueConstraint("generator_version", "title", name="kb_articles_generator_version_title"),
    )

    id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True)
    title: Mapped[str] = mapped_column(String, nullable=False)
    body: Mapped[str] = mapped_column(String, nullable=False)
    tags: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    source_kind: Mapped[str] = mapped_column(String, nullable=False)
    source_key: Mapped[str] = mapped_column(String, nullable=False)
    generator_version: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class LLMCall(Base):
    """Every OpenAI call, ever (CLAUDE.md hard rule) — the row set doubles as
    the persisted spend counter (SUM(cost_usd)) and a cache: a repeated
    (purpose, model, prompt_hash) reuses the cached response instead of
    re-billing."""

    __tablename__ = "llm_calls"
    __table_args__ = (
        UniqueConstraint("purpose", "model", "prompt_hash", name="llm_calls_cache_key"),
        Index("ix_llm_calls_purpose_created_at", "purpose", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    purpose: Mapped[str] = mapped_column(String, nullable=False)
    model: Mapped[str] = mapped_column(String, nullable=False)
    prompt_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    response: Mapped[str] = mapped_column(String, nullable=False)
    prompt_tokens: Mapped[int] = mapped_column(nullable=False)
    completion_tokens: Mapped[int] = mapped_column(nullable=False)
    cost_usd: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class EvalRun(Base):
    __tablename__ = "eval_runs"
    __table_args__ = (Index("ix_eval_runs_task_started_at", "task", "started_at"),)

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    task: Mapped[str] = mapped_column(String, nullable=False)
    model_version: Mapped[str] = mapped_column(String, nullable=False)
    dataset: Mapped[str] = mapped_column(String, nullable=False)
    split: Mapped[str] = mapped_column(String, nullable=False)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    params: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    git_sha: Mapped[str | None] = mapped_column(String(40), nullable=True)
    status: Mapped[str] = mapped_column(String, nullable=False, default="running")
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
