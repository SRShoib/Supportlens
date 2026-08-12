from typing import Literal

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    rerank: bool = True


class HighlightSpanOut(BaseModel):
    start: int
    end: int


class SearchResultOut(BaseModel):
    source: Literal["ticket", "kb_article"]
    id: str
    title: str | None
    snippet: str
    score: float
    highlights: list[HighlightSpanOut]


class SearchResponse(BaseModel):
    results: list[SearchResultOut]
    reranked: bool
