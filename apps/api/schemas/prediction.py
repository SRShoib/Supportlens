import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class PredictionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    ticket_id: uuid.UUID | None
    message_id: uuid.UUID | None
    task: str
    label: str | None
    score: float | None
    payload: dict[str, Any]
    model_version: str
    eval_run_id: uuid.UUID | None
    created_at: datetime
