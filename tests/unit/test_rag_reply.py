from unittest.mock import MagicMock

from ml.inference.base import EmbeddingResult
from ml.inference.llm_client import LLMClient
from ml.inference.rag_reply import (
    MIN_CONFIDENCE,
    build_prompt,
    draft_reply,
    extract_cited_indices,
    to_rag_sources,
)
from ml.inference.retrieval import RankedHit
from ml.inference.vector_store import VectorHit

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


def test_extract_cited_indices_finds_unique_sorted_markers() -> None:
    assert extract_cited_indices("See [2] and [1], also [2] again.") == [1, 2]


def test_extract_cited_indices_empty_when_no_markers() -> None:
    assert extract_cited_indices("No citations here.") == []


def test_to_rag_sources_uses_thread_text_for_tickets_and_document_for_kb() -> None:
    ranked = [
        RankedHit(hit=TICKET_HIT, source="ticket", score=0.9),
        RankedHit(hit=KB_HIT, source="kb_article", score=0.8),
    ]

    sources = to_rag_sources(ranked)

    assert sources[0].index == 1
    assert sources[0].kind == "ticket"
    assert sources[0].title is None
    assert sources[0].text == "Customer: my order is late\nAgent: refunded you"
    assert sources[1].index == 2
    assert sources[1].kind == "kb_article"
    assert sources[1].title == "How to Track Your Order"
    assert sources[1].text == KB_HIT.document


def test_to_rag_sources_truncates_to_max_sources() -> None:
    ranked = [RankedHit(hit=TICKET_HIT, source="ticket", score=1.0)] * 10

    assert len(to_rag_sources(ranked)) == 4


def test_build_prompt_includes_issue_and_numbered_sources() -> None:
    sources = to_rag_sources([RankedHit(hit=KB_HIT, source="kb_article", score=0.8)])

    prompt = build_prompt("my order never arrived", sources)

    assert "my order never arrived" in prompt
    assert "[1] How to Track Your Order" in prompt


def test_draft_reply_refuses_below_confidence_threshold_without_calling_llm() -> None:
    store = FakeStore({"resolved_tickets": [], "kb_articles": []})
    llm_client = MagicMock(spec=LLMClient)

    result = draft_reply(
        llm_client,
        customer_issue="totally unrelated gibberish",
        embedder=FakeEmbedder(),
        store=store,
        reranker=FakeReranker({}),
        max_tokens=400,
    )

    assert result.refused is True
    assert result.draft is None
    assert result.sources == []
    llm_client.complete.assert_not_called()


def test_draft_reply_excludes_the_target_ticket_from_its_own_retrieval_pool() -> None:
    # A ticket's Chroma id in resolved_tickets is its own ticket_id -- found
    # via a real smoke test where a ticket's own text retrieved itself as
    # the top "similar" source (docs/decisions.md). Only KB_HIT remains
    # once TICKET_HIT.id is excluded, and its score is below MIN_CONFIDENCE,
    # so this should refuse rather than draft off the excluded self-match.
    store = FakeStore({"resolved_tickets": [TICKET_HIT], "kb_articles": [KB_HIT]})
    llm_client = MagicMock(spec=LLMClient)

    result = draft_reply(
        llm_client,
        customer_issue="order is late",
        embedder=FakeEmbedder(),
        store=store,
        reranker=FakeReranker({TICKET_HIT.document: 9.0, KB_HIT.document: MIN_CONFIDENCE - 0.1}),
        max_tokens=400,
        exclude_ticket_id=TICKET_HIT.id,
    )

    assert result.refused is True
    llm_client.complete.assert_not_called()


def test_draft_reply_refuses_when_best_score_is_below_min_confidence() -> None:
    store = FakeStore({"resolved_tickets": [TICKET_HIT], "kb_articles": []})
    llm_client = MagicMock(spec=LLMClient)

    result = draft_reply(
        llm_client,
        customer_issue="order is late",
        embedder=FakeEmbedder(),
        store=store,
        reranker=FakeReranker({TICKET_HIT.document: MIN_CONFIDENCE - 0.01}),
        max_tokens=400,
    )

    assert result.refused is True
    llm_client.complete.assert_not_called()
