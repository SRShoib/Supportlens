"""Retrieval hit-rate@k (SPEC M8: "retrieval hit-rate@5 measured on a
100-query synthetic eval set"). Pure, DB-free: takes each query's already-
ranked result ids (computed by whatever retrieval variant is being
evaluated -- ml/inference/retrieval.py, dense-only or dense+rerank) plus
the known-relevant ticket id, and checks whether the relevant id appears in
the top k.
"""

from dataclasses import dataclass
from typing import Any


def hit_at_k(ranked_ids: list[str], relevant_id: str, k: int) -> bool:
    return relevant_id in ranked_ids[:k]


@dataclass(frozen=True)
class RetrievalMetrics:
    hit_rate_at_k: float
    k: int
    n_queries: int

    def to_metrics_dict(self) -> dict[str, Any]:
        return {"hit_rate_at_k": self.hit_rate_at_k, "k": self.k, "n_queries": self.n_queries}


def compute_hit_rate(
    results: list[list[str]], relevant_ids: list[str], k: int = 5
) -> RetrievalMetrics:
    if len(results) != len(relevant_ids):
        raise ValueError(
            "results and relevant_ids must be the same length (one ranked-id list per query), "
            f"got {len(results)} vs {len(relevant_ids)}"
        )
    if not results:
        return RetrievalMetrics(hit_rate_at_k=0.0, k=k, n_queries=0)

    hits = sum(
        hit_at_k(ranked, relevant, k)
        for ranked, relevant in zip(results, relevant_ids, strict=True)
    )
    return RetrievalMetrics(hit_rate_at_k=hits / len(results), k=k, n_queries=len(results))
