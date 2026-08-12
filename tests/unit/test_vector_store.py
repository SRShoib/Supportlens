from typing import Any

import pytest

from ml.inference.vector_store import ChromaVectorStore, VectorHit


class FakeCollection:
    def __init__(self) -> None:
        self.upsert_calls: list[dict[str, Any]] = []
        self._next_query_result: dict[str, Any] | None = None
        self._count = 0

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        self.upsert_calls.append(
            {"ids": ids, "embeddings": embeddings, "documents": documents, "metadatas": metadatas}
        )
        self._count = len(ids)

    def count(self) -> int:
        return self._count

    def set_query_result(self, result: dict[str, Any]) -> None:
        self._next_query_result = result

    def query(
        self,
        query_embeddings: list[list[float]],
        n_results: int,
        where: dict[str, Any] | None,
    ) -> dict[str, Any]:
        assert self._next_query_result is not None, "test must call set_query_result first"
        self.last_query_embeddings = query_embeddings
        self.last_n_results = n_results
        self.last_where = where
        return self._next_query_result


class FakeClient:
    def __init__(self) -> None:
        self.collections: dict[str, FakeCollection] = {}

    def get_or_create_collection(self, name: str, metadata: dict[str, Any]) -> FakeCollection:
        self.collections.setdefault(name, FakeCollection())
        return self.collections[name]


def test_upsert_forwards_to_the_named_collection() -> None:
    client = FakeClient()
    store = ChromaVectorStore(client=client)

    store.upsert(
        "kb_articles",
        ids=["a1"],
        embeddings=[[0.1, 0.2]],
        documents=["how to reset your password"],
        metadatas=[{"title": "Password reset"}],
    )

    assert client.collections["kb_articles"].upsert_calls == [
        {
            "ids": ["a1"],
            "embeddings": [[0.1, 0.2]],
            "documents": ["how to reset your password"],
            "metadatas": [{"title": "Password reset"}],
        }
    ]


def test_upsert_with_no_ids_is_a_noop() -> None:
    client = FakeClient()
    store = ChromaVectorStore(client=client)

    store.upsert("kb_articles", ids=[], embeddings=[], documents=[], metadatas=[])

    assert client.collections == {}


def test_query_converts_distance_to_similarity_and_zips_rows() -> None:
    client = FakeClient()
    store = ChromaVectorStore(client=client)
    collection = client.get_or_create_collection("resolved_tickets", metadata={})
    collection._count = 2
    collection.set_query_result(
        {
            "ids": [["t1", "t2"]],
            "documents": [["order is late", "refund please"]],
            "metadatas": [[{"ticket_id": "t1"}, {"ticket_id": "t2"}]],
            "distances": [[0.1, 0.4]],
        }
    )

    hits = store.query("resolved_tickets", query_embedding=[0.5, 0.5], n_results=5)

    assert hits == [
        VectorHit(id="t1", document="order is late", metadata={"ticket_id": "t1"}, similarity=0.9),
        VectorHit(
            id="t2",
            document="refund please",
            metadata={"ticket_id": "t2"},
            similarity=pytest.approx(0.6),
        ),
    ]


def test_query_caps_n_results_to_collection_count() -> None:
    client = FakeClient()
    store = ChromaVectorStore(client=client)
    collection = client.get_or_create_collection("kb_articles", metadata={})
    collection._count = 1
    collection.set_query_result(
        {
            "ids": [["a1"]],
            "documents": [["only article"]],
            "metadatas": [[{"title": "Only"}]],
            "distances": [[0.2]],
        }
    )

    hits = store.query("kb_articles", query_embedding=[0.1], n_results=5)

    assert collection.last_n_results == 1
    assert len(hits) == 1


def test_query_on_empty_collection_returns_no_hits_without_calling_query() -> None:
    client = FakeClient()
    store = ChromaVectorStore(client=client)
    client.get_or_create_collection("kb_articles", metadata={})  # count stays 0

    hits = store.query("kb_articles", query_embedding=[0.1], n_results=5)

    assert hits == []
