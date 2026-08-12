"""ROUGE-1/2/L for M6 (SPEC: "Evaluate with ROUGE-1/2/L on the transfer test
set"). Uses rouge_score.rouge_scorer directly rather than
evaluate.load("rouge") -- the `evaluate` wrapper fetches a metric script from
the HF Hub the first time it runs; this way the eval harness stays usable
offline once `rouge-score` is synced (see pyproject.toml).

SummarizationMetrics implements the same to_metrics_dict() shape
ml/evaluation/metrics.py's EvalMetrics protocol expects, so
persist_eval_run() works completely unchanged for M6's ROUGE runs -- same
reuse ml/evaluation/span_metrics.py already relies on for M4."""

from dataclasses import dataclass
from typing import Any

from rouge_score import rouge_scorer

_ROUGE_TYPES = ("rouge1", "rouge2", "rougeL")


@dataclass(frozen=True)
class SummarizationMetrics:
    rouge1: float
    rouge2: float
    # snake_case attribute (ruff N815), but the persisted/JSONB key stays
    # "rougeL" in to_metrics_dict() -- the capital-L spelling is the
    # standard ROUGE-L convention (rouge_score, HF evaluate) that any reader
    # of an eval run's metrics JSON would expect.
    rouge_l: float
    n: int

    def to_metrics_dict(self) -> dict[str, Any]:
        return {
            "rouge1": self.rouge1,
            "rouge2": self.rouge2,
            "rougeL": self.rouge_l,
            "n": self.n,
        }


def compute_rouge_metrics(predictions: list[str], references: list[str]) -> SummarizationMetrics:
    """F-measure ROUGE-1/2/L, averaged over all (prediction, reference)
    pairs -- the standard summarization-benchmark convention (stemmed, per
    rouge_scorer's default). predictions and references must be the same
    length and pair up positionally; requires at least one pair."""
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions ({len(predictions)}) and references ({len(references)}) "
            "must be the same length"
        )
    if not predictions:
        raise ValueError("compute_rouge_metrics requires at least one prediction/reference pair")

    scorer = rouge_scorer.RougeScorer(list(_ROUGE_TYPES), use_stemmer=True)
    totals = dict.fromkeys(_ROUGE_TYPES, 0.0)
    for prediction, reference in zip(predictions, references, strict=True):
        scores = scorer.score(reference, prediction)
        for rouge_type in _ROUGE_TYPES:
            totals[rouge_type] += scores[rouge_type].fmeasure

    n = len(predictions)
    return SummarizationMetrics(
        rouge1=totals["rouge1"] / n,
        rouge2=totals["rouge2"] / n,
        rouge_l=totals["rougeL"] / n,
        n=n,
    )
