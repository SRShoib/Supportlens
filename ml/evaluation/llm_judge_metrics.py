"""Aggregates the per-ticket LLM-judge scores ml/data/llm_judge_summaries.py
writes (SPEC M6: "LLM-as-judge pass ... 1-5 faithfulness/coverage rubric")
into one EvalRun-shaped summary. Implements the same to_metrics_dict() shape
ml/evaluation/metrics.py's EvalMetrics protocol expects, so
persist_eval_run() works unchanged for the judge aggregate too."""

from dataclasses import dataclass
from typing import Any, TypedDict


class JudgeRow(TypedDict):
    """The exact shape ml/data/llm_judge_summaries.py writes into each
    thread_summary_judge Prediction.payload."""

    faithfulness: int
    coverage: int
    parsed_ok: bool


@dataclass(frozen=True)
class LLMJudgeMetrics:
    n: int
    mean_faithfulness: float
    mean_coverage: float
    parsed_ok_rate: float

    def to_metrics_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "mean_faithfulness": self.mean_faithfulness,
            "mean_coverage": self.mean_coverage,
            "parsed_ok_rate": self.parsed_ok_rate,
        }


def aggregate_judge_scores(rows: list[JudgeRow]) -> LLMJudgeMetrics:
    if not rows:
        raise ValueError("aggregate_judge_scores requires at least one row")
    n = len(rows)
    return LLMJudgeMetrics(
        n=n,
        mean_faithfulness=sum(r["faithfulness"] for r in rows) / n,
        mean_coverage=sum(r["coverage"] for r in rows) / n,
        parsed_ok_rate=sum(1 for r in rows if r["parsed_ok"]) / n,
    )
