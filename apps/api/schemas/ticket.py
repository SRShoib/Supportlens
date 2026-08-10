import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from api.db.models import AuthorRole, TicketSource


class CanonicalMessage(BaseModel):
    """Loader -> DB contract. Every source loader must produce this shape."""

    id: uuid.UUID
    seq: int
    author_role: AuthorRole
    text_raw: str
    text_clean: str
    sent_at: datetime | None = None
    lang: str | None = None
    lang_confidence: float | None = None
    content_hash: str
    external_id: str
    meta: dict[str, Any] = Field(default_factory=dict)


class CanonicalTicket(BaseModel):
    id: uuid.UUID
    source: TicketSource
    external_id: str
    created_at: datetime | None = None
    channel: str
    customer_id: str | None = None
    brand: str | None = None
    lang: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)
    messages: list[CanonicalMessage]


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    ticket_id: uuid.UUID
    seq: int
    author_role: AuthorRole
    text_raw: str
    text_clean: str
    sent_at: datetime | None
    lang: str | None
    lang_confidence: float | None
    content_hash: str
    external_id: str
    meta: dict[str, Any]


class TicketOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    source: TicketSource
    external_id: str
    created_at: datetime | None
    channel: str
    customer_id: str | None
    brand: str | None
    lang: str | None
    meta: dict[str, Any]
    ingested_at: datetime
    messages: list[MessageOut] = Field(default_factory=list)
