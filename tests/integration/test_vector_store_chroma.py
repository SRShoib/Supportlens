import uuid

import pytest

from ml.inference.vector_store import ChromaVectorStore

pytestmark = pytest.mark.integration


def _collection_name() -> str:
    # Unique per test -- see chroma_store's fixture docstring for why
    # (no TRUNCATE equivalent for a session-scoped container).
    return f"test_{uuid.uuid4().hex}"


def test_upsert_and_query_round_trip_against_real_chroma(chroma_store: ChromaVectorStore) -> None:
    collection = _collection_name()

    chroma_store.upsert(
        collection,
        ids=["a", "b", "c"],
        embeddings=[[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]],
        documents=["doc a", "doc b", "doc c"],
        metadatas=[{"tag": "a"}, {"tag": "b"}, {"tag": "c"}],
    )

    hits = chroma_store.query(collection, query_embedding=[1.0, 0.0], n_results=2)

    assert [h.id for h in hits] == ["a", "c"]  # nearest, then next-nearest by cosine
    assert hits[0].document == "doc a"
    assert hits[0].metadata == {"tag": "a"}
    assert hits[0].similarity == pytest.approx(1.0, abs=1e-6)


def test_upsert_is_idempotent_by_id(chroma_store: ChromaVectorStore) -> None:
    collection = _collection_name()

    chroma_store.upsert(
        collection, ids=["a"], embeddings=[[1.0, 0.0]], documents=["first"], metadatas=[{"v": 1}]
    )
    chroma_store.upsert(
        collection, ids=["a"], embeddings=[[1.0, 0.0]], documents=["updated"], metadatas=[{"v": 2}]
    )

    assert chroma_store.count(collection) == 1
    hits = chroma_store.query(collection, query_embedding=[1.0, 0.0], n_results=5)
    assert hits[0].document == "updated"


def test_count_reflects_upserted_documents(chroma_store: ChromaVectorStore) -> None:
    collection = _collection_name()
    assert chroma_store.count(collection) == 0

    chroma_store.upsert(
        collection,
        ids=["a", "b"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        documents=["doc a", "doc b"],
        metadatas=[{"tag": "a"}, {"tag": "b"}],
    )

    assert chroma_store.count(collection) == 2


def test_query_on_a_never_upserted_collection_returns_empty_list(
    chroma_store: ChromaVectorStore,
) -> None:
    hits = chroma_store.query(_collection_name(), query_embedding=[1.0, 0.0], n_results=5)

    assert hits == []


def test_query_n_results_caps_to_collection_size(chroma_store: ChromaVectorStore) -> None:
    collection = _collection_name()
    chroma_store.upsert(
        collection,
        ids=["a"],
        embeddings=[[1.0, 0.0]],
        documents=["only doc"],
        metadatas=[{"tag": "a"}],
    )

    hits = chroma_store.query(collection, query_embedding=[1.0, 0.0], n_results=10)

    assert len(hits) == 1
