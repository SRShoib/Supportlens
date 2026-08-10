from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=100)


class TaskResultOut(BaseModel):
    label: str
    score: float
    probabilities: dict[str, float] | None = None


class PredictResponse(BaseModel):
    results: list[TaskResultOut]
