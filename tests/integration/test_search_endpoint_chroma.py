"""SPEC M8: POST /search against a REAL containerized Chroma server, not
the fake store every other search-router test uses (CLAUDE.md:
"Integration tests use testcontainers for... Chroma"). The query-time
embedder is still a fake (a fixed vector, no ML model needed) -- this
test's job is to prove the router's real chromadb.HttpClient wiring works
end-to-end, not to re-verify semantic quality (that's
scripts/generate_m8_report.py's job, against the real model and real
corpus).
"""

import pytest
from api.db import session as session_module
from api.db.session import make_engine
from api.main import app
from api.routers import search as search_router
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from ml.inference.base import EmbeddingResult
from ml.inference.vector_store import ChromaVectorStore

pytestmark = pytest.mark.integration


class FixedVectorEmbedder:
    def __init__(self, vector: list[float]) -> None:
        self._vector = vector

    def predict(self, texts: list[str]) -> list[EmbeddingResult]:
        return [EmbeddingResult(vector=self._vector) for _ in texts]


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, database_url: str) -> TestClient:
    engine = make_engine(database_url)
    monkeypatch.setattr(session_module, "engine", engine)
    monkeypatch.setattr(session_module, "SessionLocal", sessionmaker(bind=engine))
    return TestClient(app)


@pytest.fixture(autouse=True)
def _patch_search_loaders(chroma_store: ChromaVectorStore, monkeypatch: pytest.MonkeyPatch) -> None:
    # monkeypatch.setattr fully replaces (and auto-restores) these module
    # attributes for the test's duration -- no lru_cache to clear since the
    # real cached loaders are never called here.
    monkeypatch.setattr(search_router, "_get_vector_store", lambda: chroma_store)
    monkeypatch.setattr(
        search_router, "_get_embedding_predictor", lambda: FixedVectorEmbedder([1.0, 0.0])
    )


def test_search_returns_real_chroma_results_ranked_by_similarity(
    chroma_store: ChromaVectorStore, client: TestClient
) -> None:
    chroma_store.upsert(
        "kb_articles",
        ids=["near", "far"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        documents=["How to Track Your Order", "Unrelated article"],
        metadatas=[{"title": "How to Track Your Order"}, {"title": "Unrelated article"}],
    )

    response = client.post("/search", json={"query": "track my order", "rerank": False, "top_k": 5})

    assert response.status_code == 200
    results = response.json()["results"]
    assert results[0]["id"] == "near"
    assert results[0]["title"] == "How to Track Your Order"
    assert results[0]["score"] > results[1]["score"]


def test_search_combines_both_collections_from_real_chroma(
    chroma_store: ChromaVectorStore, client: TestClient
) -> None:
    chroma_store.upsert(
        "resolved_tickets",
        ids=["t1"],
        embeddings=[[1.0, 0.0]],
        documents=["order is late"],
        metadatas=[{"thread_text": "Customer: order is late\nAgent: refunded"}],
    )
    chroma_store.upsert(
        "kb_articles",
        ids=["a1"],
        embeddings=[[1.0, 0.0]],
        documents=["How to Track Your Order"],
        metadatas=[{"title": "How to Track Your Order"}],
    )

    response = client.post("/search", json={"query": "order", "rerank": False, "top_k": 5})

    assert response.status_code == 200
    sources = {r["source"] for r in response.json()["results"]}
    assert sources == {"ticket", "kb_article"}


def test_search_returns_empty_results_when_nothing_indexed(
    client: TestClient,
) -> None:
    response = client.post("/search", json={"query": "anything at all", "rerank": False})

    assert response.status_code == 200
    assert response.json()["results"] == []
