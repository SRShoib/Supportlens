from ml.inference.base import EmbeddingResult
from ml.inference.retrieval import RankedHit, retrieve
from ml.inference.vector_store import VectorHit

TICKET_HIT = VectorHit(id="t1", document="order is late", metadata={}, similarity=0.4)
KB_HIT = VectorHit(id="a1", document="track your order", metadata={}, similarity=0.9)


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


def _store() -> FakeStore:
    return FakeStore({"resolved_tickets": [TICKET_HIT], "kb_articles": [KB_HIT]})


def test_retrieve_without_reranker_ranks_by_dense_similarity() -> None:
    ranked = retrieve("order", embedder=FakeEmbedder(), store=_store(), reranker=None, pool_size=10)

    assert ranked == [
        RankedHit(hit=KB_HIT, source="kb_article", score=0.9),
        RankedHit(hit=TICKET_HIT, source="ticket", score=0.4),
    ]


def test_retrieve_with_reranker_ranks_by_cross_encoder_score() -> None:
    reranker = FakeReranker({TICKET_HIT.document: 5.0, KB_HIT.document: 1.0})

    ranked = retrieve(
        "order", embedder=FakeEmbedder(), store=_store(), reranker=reranker, pool_size=10
    )

    assert ranked == [
        RankedHit(hit=TICKET_HIT, source="ticket", score=5.0),
        RankedHit(hit=KB_HIT, source="kb_article", score=1.0),
    ]


def test_retrieve_respects_pool_size_per_collection() -> None:
    ranked = retrieve("order", embedder=FakeEmbedder(), store=_store(), reranker=None, pool_size=1)

    assert len(ranked) == 2  # 1 per collection, both collections have exactly 1 hit here


def test_retrieve_with_no_candidates_returns_empty_list() -> None:
    empty_store = FakeStore({})

    assert (
        retrieve("order", embedder=FakeEmbedder(), store=empty_store, reranker=None, pool_size=5)
        == []
    )


def test_retrieve_excludes_ids_in_exclude_ids() -> None:
    ranked = retrieve(
        "order",
        embedder=FakeEmbedder(),
        store=_store(),
        reranker=None,
        pool_size=10,
        exclude_ids=frozenset({TICKET_HIT.id}),
    )

    assert ranked == [RankedHit(hit=KB_HIT, source="kb_article", score=0.9)]
