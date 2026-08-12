"""SPEC M8: dense retrieval -> optional cross-encoder rerank -> highlighted
results, over the two Chroma collections scripts/index_search_corpus.py
writes offline. The one place in this repo where apps/api genuinely loads
an embedding model (and, when rerank=true, a cross-encoder) at request
time -- dense retrieval on an arbitrary live query can't be precomputed the
way M7's corpus embedding was, a deliberate break from the "apps/api never
loads an embedding model" precedent M7 documented (docs/decisions.md).

Both collections share one embedding space (same MiniLM checkpoint,
scripts/index_search_corpus.py), so their raw cosine similarities are
directly comparable -- ranking the combined candidate pool by similarity
(rerank=false) or cross-encoder score (rerank=true) needs no per-collection
normalization.
"""

from functools import lru_cache
from typing import Literal

from fastapi import APIRouter

from api.schemas.search import HighlightSpanOut, SearchRequest, SearchResponse, SearchResultOut
from ml.inference.base import EmbeddingPredictor
from ml.inference.highlight import highlight_matches
from ml.inference.reranker import Reranker, rerank_by_score
from ml.inference.vector_store import ChromaVectorStore, VectorHit

router = APIRouter(prefix="/search", tags=["search"])

SourceLabel = Literal["ticket", "kb_article"]

RESOLVED_TICKETS_COLLECTION = "resolved_tickets"
KB_ARTICLES_COLLECTION = "kb_articles"
# Reranking only helps if there's a wider pool to re-order -- dense
# retrieval alone just returns top_k directly per collection.
RERANK_POOL_MULTIPLIER = 4
MIN_POOL_SIZE = 10


@lru_cache
def _get_embedding_predictor() -> EmbeddingPredictor:
    # Lazy: sentence-transformers lives behind the `search` dependency
    # group, same reason apps/api/routers/predict.py defers its transformer
    # imports.
    from ml.inference.embeddings import SentenceEmbeddingPredictor

    return SentenceEmbeddingPredictor()


@lru_cache
def _get_reranker() -> Reranker:
    from ml.inference.reranker import CrossEncoderReranker

    return CrossEncoderReranker()


@lru_cache
def _get_vector_store() -> ChromaVectorStore:
    return ChromaVectorStore()


def _candidate_pool_size(top_k: int, rerank: bool) -> int:
    if not rerank:
        return max(top_k, MIN_POOL_SIZE)
    return max(top_k * RERANK_POOL_MULTIPLIER, MIN_POOL_SIZE)


def _title_for(source: SourceLabel, hit: VectorHit) -> str | None:
    if source == "kb_article":
        title = hit.metadata.get("title")
        return str(title) if title is not None else None
    return None


def _to_result(hit: VectorHit, source: SourceLabel, score: float, query: str) -> SearchResultOut:
    spans = highlight_matches(query, hit.document)
    return SearchResultOut(
        source=source,
        id=hit.id,
        title=_title_for(source, hit),
        snippet=hit.document,
        score=score,
        highlights=[HighlightSpanOut(start=s.start, end=s.end) for s in spans],
    )


@router.post("", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    store = _get_vector_store()
    embedder = _get_embedding_predictor()
    query_vector = embedder.predict([request.query])[0].vector

    pool_size = _candidate_pool_size(request.top_k, request.rerank)
    ticket_hits = store.query(RESOLVED_TICKETS_COLLECTION, query_vector, n_results=pool_size)
    kb_hits = store.query(KB_ARTICLES_COLLECTION, query_vector, n_results=pool_size)
    candidates: list[tuple[VectorHit, SourceLabel]] = [(hit, "ticket") for hit in ticket_hits] + [
        (hit, "kb_article") for hit in kb_hits
    ]

    if request.rerank and candidates:
        scores = _get_reranker().score(request.query, [hit.document for hit, _ in candidates])
    else:
        scores = [hit.similarity for hit, _ in candidates]

    ranked = rerank_by_score(list(zip(candidates, scores, strict=True)), scores)
    top = ranked[: request.top_k]

    results = [_to_result(hit, source, score, request.query) for (hit, source), score in top]
    return SearchResponse(results=results, reranked=request.rerank)
