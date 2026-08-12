from ml.inference.base import EmbeddingResult
from ml.inference.rag_reply import MIN_CONFIDENCE
from ml.inference.vector_store import VectorHit
from scripts.generate_m8_report import run_no_answer_examples

ON_TOPIC_HIT = VectorHit(id="a1", document="track your order", metadata={}, similarity=0.9)


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
        return [self._scores_by_document.get(doc, -999.0) for doc in documents]


def test_would_refuse_true_when_best_score_below_confidence() -> None:
    store = FakeStore({"resolved_tickets": [], "kb_articles": []})

    rows = run_no_answer_examples(
        [("off-topic", "gibberish query")],
        embedder=FakeEmbedder(),
        store=store,
        reranker=FakeReranker({}),
    )

    assert rows[0].would_refuse is True
    assert rows[0].best_score == float("-inf")


def test_would_refuse_false_when_best_score_at_or_above_confidence() -> None:
    store = FakeStore({"kb_articles": [ON_TOPIC_HIT]})

    rows = run_no_answer_examples(
        [("on-topic", "track my order")],
        embedder=FakeEmbedder(),
        store=store,
        reranker=FakeReranker({ON_TOPIC_HIT.document: MIN_CONFIDENCE + 1.0}),
    )

    assert rows[0].would_refuse is False
    assert rows[0].best_score == MIN_CONFIDENCE + 1.0


def test_preserves_kind_and_query_for_each_example() -> None:
    store = FakeStore({})

    rows = run_no_answer_examples(
        [("off-topic", "one"), ("on-topic", "two")],
        embedder=FakeEmbedder(),
        store=store,
        reranker=FakeReranker({}),
    )

    assert [(r.kind, r.query) for r in rows] == [("off-topic", "one"), ("on-topic", "two")]
