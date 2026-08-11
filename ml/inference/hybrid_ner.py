"""Routes each entity type to whichever system -- the rules baseline or a
trained token-classification model -- the persisted gold-set comparison
(scripts/generate_m4_report.py, docs/m4-rules-vs-model-report.md) found
actually wins for that type. Computed from real eval runs, not guessed: on
this repo's gold set, rules wins 4 of 5 entity types (ORDER_ID, PRODUCT,
DATE, ACCOUNT_REF); the transformer only clears AMOUNT. SPEC M4 predicted
this shape for ORDER_ID specifically, not across the board -- the routing
table is what actually measures it.

The routing itself lives in models/entity_routing_v1.json (written by
scripts/generate_m4_report.py alongside the report, from the same computed
comparison), not in this module -- so a re-run of the report after a new
gold set or a retrained model updates the live routing without touching
code.
"""

from collections.abc import Mapping

from ml.inference.base import EntityPredictor, EntityResult, EntitySpan


class HybridEntityPredictor:
    def __init__(
        self,
        rules_predictor: EntityPredictor,
        model_predictor: EntityPredictor,
        routing: Mapping[str, str],
    ) -> None:
        """routing maps each entity label to "rules" or "model". A label
        missing from routing defaults to "rules" -- the cheaper, always
        -available system -- rather than silently trusting an unconfigured
        model for a type nobody measured."""
        self._rules = rules_predictor
        self._model = model_predictor
        self._routing = routing

    def predict(self, texts: list[str]) -> list[EntityResult]:
        if not texts:
            return []

        rules_results = self._rules.predict(texts)
        model_results = self._model.predict(texts)

        combined: list[EntityResult] = []
        for rules_result, model_result in zip(rules_results, model_results, strict=True):
            spans: list[EntitySpan] = [
                span
                for span in rules_result.entities
                if self._routing.get(span.label, "rules") == "rules"
            ]
            spans += [
                span
                for span in model_result.entities
                if self._routing.get(span.label, "rules") == "model"
            ]
            spans.sort(key=lambda s: s.start)
            combined.append(EntityResult(entities=spans, truncated=model_result.truncated))
        return combined
