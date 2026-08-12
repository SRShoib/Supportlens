from unittest.mock import MagicMock

import pytest
from api.config import Settings
from sqlalchemy.orm import Session

from ml.inference.base import EmbeddingResult
from ml.inference.llm_client import LLMClient
from ml.inference.rag_reply import MIN_CONFIDENCE, draft_reply
from ml.inference.vector_store import VectorHit

pytestmark = pytest.mark.integration

TICKET_HIT = VectorHit(
    id="t1",
    document="my order is late",
    metadata={"thread_text": "Customer: my order is late\nAgent: refunded you"},
    similarity=0.5,
)
KB_HIT = VectorHit(
    id="a1",
    document="How to Track Your Order\ncheck order history",
    metadata={"title": "How to Track Your Order"},
    similarity=0.6,
)


class FakeEmbedder:
    def predict(self, texts: list[str]) -> list[EmbeddingResult]:
        return [EmbeddingResult(vector=[1.0, 0.0]) for _ in texts]


class FakeStore:
    def __init__(self, hits_by_collection: dict[str, list[VectorHit]]) -> None:
        self._hits_by_collection = hits_by_collection

    def query(self, collection: str, query_embedding, n_results: int = 5, where=None):
        return self._hits_by_collection.get(collection, [])[:n_results]


class FakeReranker:
    def __init__(self, scores_by_document: dict[str, float]) -> None:
        self._scores_by_document = scores_by_document

    def score(self, query: str, documents: list[str]) -> list[float]:
        return [self._scores_by_document[doc] for doc in documents]


def _mock_response(text: str) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=text))]
    response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
    return response


def _settings(**overrides: object) -> Settings:
    defaults = {"llm_enabled": True, "llm_budget_usd": 5.0, "openai_api_key": "test-key"}
    return Settings(_env_file=None, **{**defaults, **overrides})  # type: ignore[call-arg,arg-type]


def test_draft_reply_drafts_and_returns_sources_when_confident(db_session: Session) -> None:
    store = FakeStore({"resolved_tickets": [TICKET_HIT], "kb_articles": [KB_HIT]})
    openai_client = MagicMock()
    openai_client.chat.completions.create.return_value = _mock_response(
        "Refunds are on their way, see [1] and [2]."
    )
    llm_client = LLMClient(db_session, _settings(), openai_client)

    result = draft_reply(
        llm_client,
        customer_issue="order is late",
        embedder=FakeEmbedder(),
        store=store,
        reranker=FakeReranker({TICKET_HIT.document: 3.0, KB_HIT.document: 2.0}),
        max_tokens=400,
    )

    assert result.refused is False
    assert result.draft == "Refunds are on their way, see [1] and [2]."
    assert result.cited_indices == [1, 2]
    assert len(result.sources) == 2
    assert result.cached is False


def test_draft_reply_drafts_when_best_score_is_exactly_at_the_confidence_threshold(
    db_session: Session,
) -> None:
    # The gate is `score < MIN_CONFIDENCE` (ml/inference/rag_reply.py) --
    # exactly-at-threshold is confident enough, only strictly-below refuses
    # (covered separately by tests/unit/test_rag_reply.py's
    # test_draft_reply_refuses_when_best_score_is_below_min_confidence).
    store = FakeStore({"resolved_tickets": [TICKET_HIT], "kb_articles": []})
    openai_client = MagicMock()
    openai_client.chat.completions.create.return_value = _mock_response("Drafted anyway, see [1].")
    llm_client = LLMClient(db_session, _settings(), openai_client)

    result = draft_reply(
        llm_client,
        customer_issue="order is late",
        embedder=FakeEmbedder(),
        store=store,
        reranker=FakeReranker({TICKET_HIT.document: MIN_CONFIDENCE}),
        max_tokens=400,
    )

    assert result.refused is False
    assert result.draft == "Drafted anyway, see [1]."


def test_draft_reply_passes_max_tokens_through_to_the_llm_client(db_session: Session) -> None:
    store = FakeStore({"resolved_tickets": [TICKET_HIT], "kb_articles": []})
    openai_client = MagicMock()
    openai_client.chat.completions.create.return_value = _mock_response("ok")
    llm_client = LLMClient(db_session, _settings(), openai_client)

    draft_reply(
        llm_client,
        customer_issue="order is late",
        embedder=FakeEmbedder(),
        store=store,
        reranker=FakeReranker({TICKET_HIT.document: 3.0}),
        max_tokens=123,
    )

    _, kwargs = openai_client.chat.completions.create.call_args
    assert kwargs["max_tokens"] == 123


def test_repeated_draft_for_the_same_prompt_is_cached_and_not_rebilled(db_session: Session) -> None:
    store = FakeStore({"resolved_tickets": [TICKET_HIT], "kb_articles": []})
    openai_client = MagicMock()
    openai_client.chat.completions.create.return_value = _mock_response("Refunded, see [1].")
    llm_client = LLMClient(db_session, _settings(), openai_client)
    reranker = FakeReranker({TICKET_HIT.document: 3.0})

    first = draft_reply(
        llm_client,
        customer_issue="order is late",
        embedder=FakeEmbedder(),
        store=store,
        reranker=reranker,
        max_tokens=400,
    )
    second = draft_reply(
        llm_client,
        customer_issue="order is late",
        embedder=FakeEmbedder(),
        store=store,
        reranker=reranker,
        max_tokens=400,
    )

    assert first.cached is False
    assert second.cached is True
    assert second.draft == "Refunded, see [1]."
    assert openai_client.chat.completions.create.call_count == 1
