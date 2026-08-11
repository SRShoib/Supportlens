from ml.inference.base import EntityResult, EntitySpan
from ml.inference.hybrid_ner import HybridEntityPredictor


class _FakePredictor:
    def __init__(self, results: list[EntityResult]) -> None:
        self._results = results
        self.calls: list[list[str]] = []

    def predict(self, texts: list[str]) -> list[EntityResult]:
        self.calls.append(texts)
        return self._results


def test_routes_each_label_to_the_configured_system() -> None:
    text = "order ORD-99321 charged $49.99"
    rules = _FakePredictor(
        [
            EntityResult(
                entities=[
                    EntitySpan(6, 15, "ORDER_ID", "ORD-99321", score=1.0),
                    EntitySpan(24, 30, "AMOUNT", "wrong!", score=1.0),
                ]
            )
        ]
    )
    model = _FakePredictor(
        [
            EntityResult(
                entities=[
                    EntitySpan(6, 15, "ORDER_ID", "wrong-model-guess", score=0.4),
                    EntitySpan(24, 30, "AMOUNT", "$49.99", score=0.9),
                ]
            )
        ]
    )
    routing = {"ORDER_ID": "rules", "AMOUNT": "model"}

    predictor = HybridEntityPredictor(rules, model, routing)
    results = predictor.predict([text])

    assert len(results) == 1
    labels_and_text = {(s.label, s.text) for s in results[0].entities}
    assert labels_and_text == {("ORDER_ID", "ORD-99321"), ("AMOUNT", "$49.99")}


def test_label_missing_from_routing_defaults_to_rules() -> None:
    rules = _FakePredictor([EntityResult(entities=[EntitySpan(0, 4, "DATE", "text", score=1.0)])])
    model = _FakePredictor([EntityResult(entities=[EntitySpan(0, 4, "DATE", "text", score=1.0)])])

    predictor = HybridEntityPredictor(rules, model, routing={})  # DATE unconfigured
    results = predictor.predict(["text"])

    assert len(results[0].entities) == 1
    assert results[0].entities[0].label == "DATE"


def test_both_predictors_called_with_the_same_texts() -> None:
    rules = _FakePredictor([EntityResult(entities=[]), EntityResult(entities=[])])
    model = _FakePredictor([EntityResult(entities=[]), EntityResult(entities=[])])
    predictor = HybridEntityPredictor(rules, model, routing={})

    predictor.predict(["a", "b"])

    assert rules.calls == [["a", "b"]]
    assert model.calls == [["a", "b"]]


def test_truncated_flag_comes_from_the_model_result() -> None:
    rules = _FakePredictor([EntityResult(entities=[], truncated=False)])
    model = _FakePredictor([EntityResult(entities=[], truncated=True)])
    predictor = HybridEntityPredictor(rules, model, routing={})

    results = predictor.predict(["text"])

    assert results[0].truncated is True


def test_empty_texts_list_short_circuits_without_calling_either_predictor() -> None:
    rules = _FakePredictor([])
    model = _FakePredictor([])
    predictor = HybridEntityPredictor(rules, model, routing={})

    assert predictor.predict([]) == []
    assert rules.calls == []
    assert model.calls == []


def test_spans_are_sorted_by_start_regardless_of_source() -> None:
    rules = _FakePredictor(
        [EntityResult(entities=[EntitySpan(20, 25, "ORDER_ID", "later", score=1.0)])]
    )
    model = _FakePredictor(
        [EntityResult(entities=[EntitySpan(0, 5, "AMOUNT", "first", score=1.0)])]
    )
    routing = {"ORDER_ID": "rules", "AMOUNT": "model"}

    predictor = HybridEntityPredictor(rules, model, routing)
    results = predictor.predict(["some text here for testing spans"])

    assert [s.text for s in results[0].entities] == ["first", "later"]


def test_offset_invariant_holds_for_every_merged_span() -> None:
    text = "order ORD-99321 charged $49.99"
    rules = _FakePredictor(
        [EntityResult(entities=[EntitySpan(6, 15, "ORDER_ID", "ORD-99321", score=1.0)])]
    )
    model = _FakePredictor(
        [EntityResult(entities=[EntitySpan(24, 30, "AMOUNT", "$49.99", score=0.9)])]
    )
    routing = {"ORDER_ID": "rules", "AMOUNT": "model"}

    predictor = HybridEntityPredictor(rules, model, routing)
    results = predictor.predict([text])

    for span in results[0].entities:
        assert text[span.start : span.end] == span.text
