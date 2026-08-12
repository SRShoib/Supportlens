"""SPEC M8: POST /tickets/{id}/suggested-reply -- a RAG-drafted suggested
reply for a support agent, retrieval-grounded and cited
(ml/inference/rag_reply.py). The query embedded for retrieval is the
target ticket's own customer-problem text (the same document unit
scripts/index_search_corpus.py builds for every *other* ticket) -- "find
cases like this one".

Deliberately not persisted as a Prediction, unlike M5/M6's per-ticket
aggregates: a suggested reply is interactive/on-demand, not a stable fact
computed once by an offline backfill script, so it stays live like
/predict/summary. The `llm_calls` cache underneath LLMClient still means a
second view of the same ticket doesn't re-bill (docs/decisions.md).
"""

from functools import lru_cache
from uuid import UUID

from fastapi import APIRouter, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from api.config import Settings
from api.db.models import Ticket
from api.deps import DbDep, SettingsDep
from api.schemas.rag import RagSourceOut, SuggestedReplyResponse
from ml.inference.base import EmbeddingPredictor
from ml.inference.llm_client import BudgetExceededError, LLMClient, LLMDisabledError
from ml.inference.rag_reply import draft_reply
from ml.inference.reranker import Reranker
from ml.inference.vector_store import ChromaVectorStore
from scripts.compute_embeddings import build_documents

router = APIRouter(prefix="/tickets", tags=["rag"])


@lru_cache
def _get_embedding_predictor() -> EmbeddingPredictor:
    from ml.inference.embeddings import SentenceEmbeddingPredictor

    return SentenceEmbeddingPredictor()


@lru_cache
def _get_reranker() -> Reranker:
    from ml.inference.reranker import CrossEncoderReranker

    return CrossEncoderReranker()


@lru_cache
def _get_vector_store() -> ChromaVectorStore:
    return ChromaVectorStore()


def _build_llm_client(db: Session, settings: Settings) -> LLMClient:
    # Not lru_cache'd like the model loaders above -- LLMClient is bound to
    # a request-scoped db Session, a new one every request. Its own
    # indirection (rather than calling LLMClient(db, settings) inline
    # below) exists purely so tests can monkeypatch this one function to
    # inject a mocked openai_client (CLAUDE.md: never hit a paid API from
    # tests), the same swap-a-loader pattern the three functions above use
    # for heavy models.
    return LLMClient(db, settings)


@router.post("/{ticket_id}/suggested-reply", response_model=SuggestedReplyResponse)
def suggested_reply(ticket_id: UUID, db: DbDep, settings: SettingsDep) -> SuggestedReplyResponse:
    stmt = select(Ticket).options(selectinload(Ticket.messages)).where(Ticket.id == ticket_id)
    ticket = db.scalars(stmt).first()
    if ticket is None:
        raise HTTPException(status_code=404, detail="ticket not found")

    _, documents = build_documents([ticket])
    if not documents:
        raise HTTPException(
            status_code=422, detail="ticket has no customer message to draft a reply for"
        )

    llm_client = _build_llm_client(db, settings)
    try:
        result = draft_reply(
            llm_client,
            customer_issue=documents[0],
            embedder=_get_embedding_predictor(),
            store=_get_vector_store(),
            reranker=_get_reranker(),
            max_tokens=settings.rag_max_completion_tokens,
            exclude_ticket_id=str(ticket_id),
        )
    except LLMDisabledError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except BudgetExceededError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return SuggestedReplyResponse(
        refused=result.refused,
        refusal_reason=result.refusal_reason,
        draft=result.draft,
        cited_indices=result.cited_indices,
        cached=result.cached,
        cost_usd=result.cost_usd,
        sources=[
            RagSourceOut(
                index=s.index, kind=s.kind, id=s.id, title=s.title, text=s.text, score=s.score
            )
            for s in result.sources
        ],
    )
