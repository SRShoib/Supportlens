import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class EvalRunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    task: str
    model_version: str
    dataset: str
    split: str
    metrics: dict[str, Any]
    params: dict[str, Any]
    git_sha: str | None
    status: str
    started_at: datetime
    finished_at: datetime | None
