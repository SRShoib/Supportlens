"""Thin Chroma client wrapper (SPEC M8: "Index resolved tickets + a small
synthetic KB... in Chroma"). apps/api loads this at request time to embed a
live query and query Chroma directly -- unlike M7, whose
ml/inference/embeddings.py docstring says apps/api "never loads an embedding
model" at all. M8 breaks that precedent out of necessity: dense retrieval on
an arbitrary live query can't be precomputed offline the way M7's corpus
embedding was (docs/decisions.md).

Two collections, both written offline by scripts/index_search_corpus.py and
only ever read live by apps/api/routers/search.py and rag.py:
  - "resolved_tickets": one row per resolved ticket (SPEC M8's "resolved
    cases"). `document` is the ticket's customer-problem text (the same
    text that was embedded, so a search snippet highlights against exactly
    what matched); `metadata["thread_text"]` carries the full thread
    (including the agent's resolution) for the RAG step to cite from.
  - "kb_articles": one row per ml/data/kb_generate.py article. `document` is
    "{title}\\n{body}"; metadata carries title/tags separately for display.

Deliberately DB-agnostic (no api.db.models import, same boundary
ml/inference/sentiment_trajectory.py keeps) and unit-testable without
chromadb installed: the `chromadb` import only happens inside __init__'s
`client is None` branch, never at module level, so a test that constructs
this class with a fake injected client never touches it -- same lazy-import
convention apps/api/routers/predict.py uses for transformers/torch.
"""

from dataclasses import dataclass
from typing import Any, Protocol

# cosine distance is unbounded above 1 in general, but for the normalized
# embeddings sentence-transformers produces it falls in [0, 2]; similarity
# =1-distance therefore falls in [-1, 1], same range as a raw cosine score.
COSINE_SPACE = {"hnsw:space": "cosine"}


@dataclass(frozen=True)
class VectorHit:
    id: str
    document: str
    metadata: dict[str, Any]
    similarity: float


class Collection(Protocol):
    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None: ...

    def query(
        self,
        query_embeddings: list[list[float]],
        n_results: int,
        where: dict[str, Any] | None,
    ) -> dict[str, Any]: ...

    def count(self) -> int: ...


class ChromaClient(Protocol):
    # `metadata` is keyword-only here (unlike Collection.upsert/query above)
    # so this stays structurally compatible with chromadb's real ClientAPI,
    # whose get_or_create_collection has extra positional params
    # (configuration, embedding_function, data_loader) before its own
    # metadata kwarg -- this class is only ever called with metadata= (see
    # _collection below), never positionally.
    def get_or_create_collection(self, name: str, *, metadata: dict[str, Any]) -> Collection: ...


class ChromaVectorStore:
    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        client: ChromaClient | None = None,
    ) -> None:
        if client is not None:
            self._client = client
        else:
            import chromadb
            from api.config import get_settings

            settings = get_settings()
            # chromadb.HttpClient/PersistentClient's real return type
            # (ClientAPI) is a much wider, numpy/generic-parameterized
            # interface than the minimal ChromaClient Protocol above --
            # structurally compatible at runtime (every call site here uses
            # metadata= by keyword) but not close enough for mypy's
            # protocol-assignment check to verify on its own.
            if settings.chroma_embedded_path:
                self._client = chromadb.PersistentClient(  # type: ignore[assignment]
                    path=settings.chroma_embedded_path
                )
            else:
                self._client = chromadb.HttpClient(  # type: ignore[assignment]
                    host=host or settings.chroma_host, port=port or settings.chroma_port
                )

    def _collection(self, name: str) -> Collection:
        return self._client.get_or_create_collection(name, metadata=COSINE_SPACE)

    def upsert(
        self,
        collection: str,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        if not ids:
            return
        self._collection(collection).upsert(
            ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
        )

    def count(self, collection: str) -> int:
        return self._collection(collection).count()

    def query(
        self,
        collection: str,
        query_embedding: list[float],
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[VectorHit]:
        col = self._collection(collection)
        n_available = col.count()
        if n_available == 0:
            return []
        result = col.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, n_available),
            where=where,
        )
        ids = result["ids"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]
        return [
            VectorHit(id=id_, document=doc, metadata=meta, similarity=1.0 - dist)
            for id_, doc, meta, dist in zip(ids, documents, metadatas, distances, strict=True)
        ]
