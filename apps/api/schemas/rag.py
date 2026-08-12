from pydantic import BaseModel


class RagSourceOut(BaseModel):
    index: int
    kind: str
    id: str
    title: str | None
    text: str
    score: float


class SuggestedReplyResponse(BaseModel):
    refused: bool
    refusal_reason: str | None
    draft: str | None
    cited_indices: list[int]
    cached: bool
    cost_usd: float
    sources: list[RagSourceOut]
