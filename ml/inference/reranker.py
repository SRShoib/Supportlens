"""Cross-encoder reranker (SPEC M8: "optional cross-encoder rerank
(ms-marco-MiniLM)"). Scores a query against each candidate document
directly in one forward pass per pair -- more accurate than the dense
bi-encoder's cosine similarity, but too slow to run over a whole
collection, so it's only ever a second pass over dense retrieval's top-N
candidates, never the first-pass retrieval mechanism itself.

Loaded lazily by apps/api/routers/search.py (same lazy-import convention
apps/api/routers/predict.py uses for its transformer predictors), so a
dense-only deployment never needs the cross-encoder checkpoint downloaded.
"""

from typing import Protocol, TypeVar

DEFAULT_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

T = TypeVar("T")


class Reranker(Protocol):
    """apps/api/routers/search.py's lazy-loader return type -- lets it
    type-check without importing CrossEncoderReranker (and therefore
    sentence_transformers) at module level, same boundary
    ml/inference/base.py's Predictor protocol keeps for apps/api/routers/predict.py."""

    def score(self, query: str, documents: list[str]) -> list[float]: ...


def rerank_by_score(items: list[T], scores: list[float]) -> list[T]:
    """Descending sort of `items` by their paired `scores` (same index in
    both lists). Stable: equal scores keep their original relative order."""
    if len(items) != len(scores):
        raise ValueError(
            f"items and scores must be the same length, got {len(items)} vs {len(scores)}"
        )
    ranked = sorted(range(len(items)), key=lambda i: scores[i], reverse=True)
    return [items[i] for i in ranked]


class CrossEncoderReranker:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)

    def score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        pairs = [(query, document) for document in documents]
        scores = self._model.predict(pairs)
        return [float(s) for s in scores]
