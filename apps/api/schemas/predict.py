from typing import Literal

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=100)
    model: Literal["baseline", "transformer"] = "baseline"


class TaskResultOut(BaseModel):
    label: str
    score: float
    probabilities: dict[str, float] | None = None


class PredictResponse(BaseModel):
    results: list[TaskResultOut]


class EntitySpanOut(BaseModel):
    start: int
    end: int
    label: str
    text: str
    score: float


class EntityResultOut(BaseModel):
    entities: list[EntitySpanOut]
    truncated: bool = False


class EntitiesResponse(BaseModel):
    results: list[EntityResultOut]
