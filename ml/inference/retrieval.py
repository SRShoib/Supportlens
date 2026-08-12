"""Shared dense-retrieval-plus-optional-rerank pipeline (SPEC M8), used by
both apps/api/routers/search.py (the interactive search endpoint) and
ml/inference/rag_reply.py (suggested-reply drafting): embed the query once,
pull a candidate pool from both Chroma collections, and rank it either by
raw cosine similarity (reranker=None) or by cross-encoder score. Factored
out so the two callers can never drift into two slightly different ranking
implementations -- CLAUDE.md: "model choice... selected by config/request
flag, not code duplication" applies here even though this isn't a
baseline/transformer choice.
"""

from dataclasses import dataclass
from typing import Literal

from ml.inference.base import EmbeddingPredictor
from ml.inference.reranker import Reranker, rerank_by_score
from ml.inference.vector_store import ChromaVectorStore, VectorHit

SourceLabel = Literal["ticket", "kb_article"]

RESOLVED_TICKETS_COLLECTION = "resolved_tickets"
KB_ARTICLES_COLLECTION = "kb_articles"


@dataclass(frozen=True)
class RankedHit:
    hit: VectorHit
    source: SourceLabel
    score: float


def retrieve(
    query: str,
    *,
    embedder: EmbeddingPredictor,
    store: ChromaVectorStore,
    reranker: Reranker | None,
    pool_size: int,
    exclude_ids: frozenset[str] = frozenset(),
) -> list[RankedHit]:
    """Both collections share one embedding space (same MiniLM checkpoint,
    scripts/index_search_corpus.py), so their raw cosine similarities are
    directly comparable -- ranking the combined pool needs no
    per-collection normalization either way.

    `exclude_ids` matters for ml/inference/rag_reply.py: a ticket's Chroma
    id in `resolved_tickets` is its own ticket_id, so drafting a reply for
    ticket X by querying with X's own customer-problem text would otherwise
    retrieve X itself as a near-perfect (identical-text) match every time --
    found via a real smoke-test call, not a test fixture (docs/decisions.md).
    apps/api/routers/search.py never has anything to exclude, so its calls
    just use the default empty set."""
    query_vector = embedder.predict([query])[0].vector
    ticket_hits = store.query(RESOLVED_TICKETS_COLLECTION, query_vector, n_results=pool_size)
    kb_hits = store.query(KB_ARTICLES_COLLECTION, query_vector, n_results=pool_size)
    candidates: list[tuple[VectorHit, SourceLabel]] = [
        (hit, "ticket") for hit in ticket_hits if hit.id not in exclude_ids
    ] + [(hit, "kb_article") for hit in kb_hits if hit.id not in exclude_ids]
    if not candidates:
        return []

    if reranker is not None:
        scores = reranker.score(query, [hit.document for hit, _ in candidates])
    else:
        scores = [hit.similarity for hit, _ in candidates]

    ranked_pairs = rerank_by_score(list(zip(candidates, scores, strict=True)), scores)
    return [RankedHit(hit=hit, source=source, score=score) for (hit, source), score in ranked_pairs]
