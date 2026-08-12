import pytest
from api.main import app
from api.routers import search
from fastapi.testclient import TestClient

from ml.inference.base import EmbeddingResult
from ml.inference.vector_store import VectorHit

client = TestClient(app)


class FakeEmbeddingPredictor:
    def predict(self, texts: list[str]) -> list[EmbeddingResult]:
        return [EmbeddingResult(vector=[1.0, 0.0]) for _ in texts]


class FakeVectorStore:
    def __init__(self, hits_by_collection: dict[str, list[VectorHit]]) -> None:
        self._hits_by_collection = hits_by_collection

    def query(
        self, collection: str, query_embedding: list[float], n_results: int = 5, where=None
    ) -> list[VectorHit]:
        return self._hits_by_collection.get(collection, [])[:n_results]


class FakeReranker:
    def __init__(self, scores_by_id: dict[str, float]) -> None:
        self._scores_by_id = scores_by_id

    def score(self, query: str, documents: list[str]) -> list[float]:
        # documents don't carry an id here, so tests key the fake by
        # document text instead -- simpler than plumbing ids through.
        return [self._scores_by_id[doc] for doc in documents]


TICKET_HIT = VectorHit(
    id="t1", document="order is late", metadata={"ticket_id": "t1"}, similarity=0.4
)
KB_HIT = VectorHit(
    id="a1",
    document="How to Track Your Order\ncheck order history",
    metadata={"title": "How to Track Your Order"},
    similarity=0.9,
)


@pytest.fixture(autouse=True)
def _reset_caches() -> None:
    yield
    search._get_embedding_predictor.cache_clear()
    search._get_vector_store.cache_clear()
    search._get_reranker.cache_clear()


def _patch_no_rerank(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(search, "_get_embedding_predictor", lambda: FakeEmbeddingPredictor())
    monkeypatch.setattr(
        search,
        "_get_vector_store",
        lambda: FakeVectorStore({"resolved_tickets": [TICKET_HIT], "kb_articles": [KB_HIT]}),
    )


def test_search_without_rerank_sorts_by_similarity_descending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_no_rerank(monkeypatch)

    response = client.post("/search", json={"query": "order", "rerank": False})

    assert response.status_code == 200
    body = response.json()
    assert body["reranked"] is False
    assert [r["id"] for r in body["results"]] == ["a1", "t1"]
    assert body["results"][0]["score"] == pytest.approx(0.9)


def test_search_with_rerank_reorders_using_cross_encoder_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_no_rerank(monkeypatch)
    # Cross-encoder disagrees with dense similarity: the ticket (lower
    # cosine similarity) scores higher here, proving rerank=true actually
    # changes the ranking rather than just relabeling it.
    monkeypatch.setattr(
        search,
        "_get_reranker",
        lambda: FakeReranker({TICKET_HIT.document: 5.0, KB_HIT.document: 1.0}),
    )

    response = client.post("/search", json={"query": "order", "rerank": True})

    assert response.status_code == 200
    body = response.json()
    assert body["reranked"] is True
    assert [r["id"] for r in body["results"]] == ["t1", "a1"]
    assert body["results"][0]["score"] == pytest.approx(5.0)


def test_search_respects_top_k(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_no_rerank(monkeypatch)

    response = client.post("/search", json={"query": "order", "rerank": False, "top_k": 1})

    assert response.status_code == 200
    assert len(response.json()["results"]) == 1


def test_kb_result_includes_title_ticket_result_does_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_no_rerank(monkeypatch)

    response = client.post("/search", json={"query": "order", "rerank": False})

    results_by_id = {r["id"]: r for r in response.json()["results"]}
    assert results_by_id["a1"]["title"] == "How to Track Your Order"
    assert results_by_id["a1"]["source"] == "kb_article"
    assert results_by_id["t1"]["title"] is None
    assert results_by_id["t1"]["source"] == "ticket"


def test_highlights_mark_query_term_matches_in_the_snippet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_no_rerank(monkeypatch)

    response = client.post("/search", json={"query": "order", "rerank": False})

    ticket_result = next(r for r in response.json()["results"] if r["id"] == "t1")
    assert len(ticket_result["highlights"]) == 1
    span = ticket_result["highlights"][0]
    assert ticket_result["snippet"][span["start"] : span["end"]] == "order"


def test_rejects_empty_query() -> None:
    response = client.post("/search", json={"query": ""})
    assert response.status_code == 422


def test_rejects_top_k_out_of_range() -> None:
    response = client.post("/search", json={"query": "order", "top_k": 0})
    assert response.status_code == 422
