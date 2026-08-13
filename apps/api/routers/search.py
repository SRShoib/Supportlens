"""SPEC M8: dense retrieval -> optional cross-encoder rerank -> highlighted
results, over the two Chroma collections scripts/index_search_corpus.py
writes offline. The one place in this repo where apps/api genuinely loads
an embedding model (and, when rerank=true, a cross-encoder) at request
time -- dense retrieval on an arbitrary live query can't be precomputed the
way M7's corpus embedding was, a deliberate break from the "apps/api never
loads an embedding model" precedent M7 documented (docs/decisions.md).

The actual embed -> retrieve -> rank pipeline lives in
ml/inference/retrieval.py, shared with ml/inference/rag_reply.py -- this
module is just the HTTP shape around it (request validation, model-loader
caching, highlighting for display).
"""

from functools import lru_cache

from fastapi import APIRouter, HTTPException

from api.deps import SettingsDep
from api.schemas.search import HighlightSpanOut, SearchRequest, SearchResponse, SearchResultOut
from ml.inference.base import EmbeddingPredictor
from ml.inference.highlight import highlight_matches
from ml.inference.reranker import Reranker
from ml.inference.retrieval import RankedHit, SourceLabel, retrieve
from ml.inference.vector_store import ChromaVectorStore, VectorHit

router = APIRouter(prefix="/search", tags=["search"])

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


def _to_result(ranked: RankedHit, query: str) -> SearchResultOut:
    spans = highlight_matches(query, ranked.hit.document)
    return SearchResultOut(
        source=ranked.source,
        id=ranked.hit.id,
        title=_title_for(ranked.source, ranked.hit),
        snippet=ranked.hit.document,
        score=ranked.score,
        highlights=[HighlightSpanOut(start=s.start, end=s.end) for s in spans],
    )


@router.post("", response_model=SearchResponse)
def search(request: SearchRequest, settings: SettingsDep) -> SearchResponse:
    if not settings.search_enabled:
        # Checked before any of the lazy loaders below run -- the point is
        # to never import sentence-transformers/torch at all on a host
        # where SEARCH_ENABLED=false, not just to fail after loading them.
        raise HTTPException(
            status_code=503,
            detail="Search is not available on this deployment (SEARCH_ENABLED=false).",
        )
    pool_size = _candidate_pool_size(request.top_k, request.rerank)
    ranked = retrieve(
        request.query,
        embedder=_get_embedding_predictor(),
        store=_get_vector_store(),
        reranker=_get_reranker() if request.rerank else None,
        pool_size=pool_size,
    )
    top = ranked[: request.top_k]
    results = [_to_result(r, request.query) for r in top]
    return SearchResponse(results=results, reranked=request.rerank)
