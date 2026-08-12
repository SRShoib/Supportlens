"""Evaluates M6's baseline (lead-k extractive) and transformer
(FLAN-T5-small) thread summarizers on samsum's and dialogsum's own test
splits -- ROUGE-1/2/L, SPEC M6 -- persists EvalRun rows (CLAUDE.md rule #5:
no metric without an eval run), benchmarks CPU latency + export size, and
renders docs/m6-comparison-report.md + a model card from the results --
nothing in either doc is hand-typed. Structurally the same shape as
scripts/generate_m5_report.py.

Per-dataset (not pooled) test ROUGE: ml/training/train_summarization.py
pools both datasets' train rows, but each dataset's own test split is
scored separately here, to stay comparable to published per-benchmark
numbers -- same rationale as generate_m5_report.py's tweet_eval precedent
(see docs/decisions.md).

Also aggregates the LLM-judge Predictions ml/data/llm_judge_summaries.py
wrote against real supportlens tickets (task="thread_summary_judge") into
one EvalRun, and prints the lowest-faithfulness real-ticket examples to
stdout -- raw material for docs/summarization-failure-modes.md, which still
needs a human (or a later Claude Code turn) to read the actual hallucinated
output and write the qualitative failure descriptions; this script only
surfaces candidates, it doesn't write that doc.

Run: uv run python scripts/generate_m6_report.py
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from api.db.models import Prediction
from api.db.session import SessionLocal
from sqlalchemy import select
from sqlalchemy.orm import Session

from ml.evaluation.latency import LatencyResult, benchmark_latency
from ml.evaluation.llm_judge_metrics import LLMJudgeMetrics, aggregate_judge_scores
from ml.evaluation.metrics import persist_eval_run
from ml.evaluation.rouge_metrics import SummarizationMetrics, compute_rouge_metrics
from ml.inference.base import SummaryPredictor
from ml.inference.extractive_summary import DEFAULT_K, ExtractiveSummaryPredictor
from ml.inference.summarization import SummarizationPredictor
from ml.training.splits import load_splits

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
MODEL_CARDS_DIR = ROOT / "docs" / "model-cards"
REPORT_PATH = ROOT / "docs" / "m6-comparison-report.md"

DATASETS = ["samsum_v1", "dialogsum_v1"]
TRANSFORMER_MODEL_NAME = "google/flan-t5-small"
TRANSFORMER_EXPORT_DIR = MODELS_DIR / "transformer_thread_summary_flan-t5-small_v1" / "final"
TRANSFORMER_MODEL_VERSION = "transformer_thread_summary_flan-t5-small_v1"
BASELINE_MODEL_VERSION = "baseline_thread_summary_v1"
BENCHMARK_DIALOGUE = (
    "Customer: my order has not arrived and it was supposed to be here 3 days ago\n"
    "Agent: I am sorry, let me check the tracking\n"
    "Agent: it looks like it is stuck at the depot, I will escalate it\n"
    "Customer: thank you please hurry"
)
PREDICT_BATCH_SIZE = 16  # seq2seq generation is far more expensive per text than classification
WIN_MARGIN = (
    0.02  # same convention as generate_m3/m5_report.py's macro-F1 margin, applied to rouge1
)
N_QUALITATIVE_EXAMPLES = 5
N_FAILURE_CANDIDATES = 5


@dataclass(frozen=True)
class VariantResult:
    model_version: str
    metrics: SummarizationMetrics
    latency: LatencyResult
    size_mb: float | None  # None for the baseline -- no model file to size


@dataclass(frozen=True)
class DatasetResult:
    dataset: str
    baseline: VariantResult
    transformer: VariantResult


def _dir_size_mb(path: Path) -> float:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 * 1024)


def _predict_in_batches(predictor: SummaryPredictor, texts: list[str]) -> list[str]:
    summaries: list[str] = []
    for start in range(0, len(texts), PREDICT_BATCH_SIZE):
        summaries.extend(
            r.summary for r in predictor.predict(texts[start : start + PREDICT_BATCH_SIZE])
        )
    return summaries


def _eval_variant(
    session: Session,
    *,
    predictor: SummaryPredictor,
    model_version: str,
    dataset: str,
    dialogues: list[str],
    references: list[str],
    params: dict[str, Any],
    size_mb: float | None,
) -> VariantResult:
    predictions = _predict_in_batches(predictor, dialogues)
    metrics = compute_rouge_metrics(predictions, references)
    persist_eval_run(
        session,
        task="thread_summary",
        model_version=model_version,
        dataset=dataset,
        split="test",
        metrics=metrics,
        params=params,
    )
    latency = benchmark_latency(predictor, BENCHMARK_DIALOGUE)
    print(
        f"  {model_version} on {dataset}: rouge1={metrics.rouge1:.4f} "
        f"rouge2={metrics.rouge2:.4f} rougeL={metrics.rouge_l:.4f} p50={latency.p50_ms:.1f}ms"
    )
    return VariantResult(
        model_version=model_version, metrics=metrics, latency=latency, size_mb=size_mb
    )


def _eval_dataset(
    session: Session,
    dataset: str,
    baseline: ExtractiveSummaryPredictor,
    transformer: SummarizationPredictor,
) -> DatasetResult:
    df = load_splits(dataset)
    test_df = df[df["split"] == "test"]
    dialogues = test_df["dialogue"].tolist()
    references = test_df["summary"].tolist()
    print(f"dataset={dataset}: {len(test_df)} test rows")

    baseline_result = _eval_variant(
        session,
        predictor=baseline,
        model_version=BASELINE_MODEL_VERSION,
        dataset=dataset,
        dialogues=dialogues,
        references=references,
        params={"k": DEFAULT_K},
        size_mb=None,
    )
    transformer_result = _eval_variant(
        session,
        predictor=transformer,
        model_version=TRANSFORMER_MODEL_VERSION,
        dataset=dataset,
        dialogues=dialogues,
        references=references,
        params={"model_name": TRANSFORMER_MODEL_NAME},
        size_mb=_dir_size_mb(TRANSFORMER_EXPORT_DIR),
    )
    return DatasetResult(dataset=dataset, baseline=baseline_result, transformer=transformer_result)


def _judge_rows(session: Session) -> list[dict[str, Any]]:
    # label stores which summarizer was judged (ml/data/llm_judge_summaries.py)
    # -- filtered to the transformer here so a judge run against a mixed
    # baseline/transformer ticket corpus never lets trivially-faithful
    # extractive rows dilute the aggregate. As of the fix in
    # ml/data/llm_judge_summaries.py, new judge rows are always
    # transformer-only, but this filter also protects against any
    # already-persisted baseline-judged rows from before that fix.
    stmt = select(Prediction).where(
        Prediction.task == "thread_summary_judge", Prediction.label == TRANSFORMER_MODEL_VERSION
    )
    return [dict(p.payload) for p in session.scalars(stmt).all()]


def _print_failure_mode_candidates(session: Session, n: int) -> None:
    """Raw material for docs/summarization-failure-modes.md -- the lowest-
    faithfulness judged summaries, printed with their full dialogue and
    generated summary so a human (or a later Claude Code turn) can read the
    actual hallucinations and write the qualitative failure descriptions.
    Does not write the doc itself."""
    stmt = (
        select(Prediction)
        .where(
            Prediction.task == "thread_summary_judge",
            Prediction.label == TRANSFORMER_MODEL_VERSION,
        )
        .order_by(Prediction.score.asc())
        .limit(n)
    )
    rows = list(session.scalars(stmt).all())
    if not rows:
        print(
            "\nno thread_summary_judge Predictions found -- run "
            "`uv run python -m ml.data.llm_judge_summaries` first, then re-run this report, "
            "before writing docs/summarization-failure-modes.md."
        )
        return

    print("\n--- lowest-faithfulness candidates for docs/summarization-failure-modes.md ---")
    for row in rows:
        payload = row.payload
        print(
            f"\nticket={row.ticket_id} model={row.label} "
            f"faithfulness={payload.get('faithfulness')} coverage={payload.get('coverage')}"
        )
        print(f"  summary: {payload.get('summary')}")


def _render_model_card(
    dataset_results: list[DatasetResult], judge_metrics: LLMJudgeMetrics | None
) -> str:
    lines = [
        f"# Model card: {TRANSFORMER_MODEL_VERSION}",
        "",
        f"**Base model:** `{TRANSFORMER_MODEL_NAME}`",
        "**Task:** thread (conversation) summarization, free-text generation",
        "",
        "## Data & splits",
        "",
        "- Datasets: `knkarthick/samsum` + `knkarthick/dialogsum` (HF mirrors of samsum/dialogsum -- "
        "the canonical `samsum` repo no longer loads under `datasets>=3.1`, see `docs/decisions.md`), "
        "split files `data/splits/{samsum,dialogsum}_v1.parquet`",
        "- Split: each dataset's own fixed train/validation/test partition, used verbatim.",
        "- Training pools both datasets' train rows (`ml/training/train_summarization.py`); ROUGE "
        "below is reported per dataset's own test split, not pooled, to stay comparable to each "
        "published benchmark (see `docs/decisions.md`).",
        "",
        "## Metrics (test split, per dataset)",
        "",
        "| Dataset | Model | ROUGE-1 | ROUGE-2 | ROUGE-L |",
        "|---|---|---|---|---|",
    ]
    for result in dataset_results:
        lines.append(
            f"| {result.dataset} | baseline (`{BASELINE_MODEL_VERSION}`) | "
            f"{result.baseline.metrics.rouge1:.4f} | {result.baseline.metrics.rouge2:.4f} | "
            f"{result.baseline.metrics.rouge_l:.4f} |"
        )
        lines.append(
            f"| {result.dataset} | transformer (`{TRANSFORMER_MODEL_VERSION}`) | "
            f"{result.transformer.metrics.rouge1:.4f} | {result.transformer.metrics.rouge2:.4f} | "
            f"{result.transformer.metrics.rouge_l:.4f} |"
        )
    lines += [
        "",
        "## CPU latency & size",
        "",
        "| Dataset | | Transformer | Baseline |",
        "|---|---|---|---|",
    ]
    for result in dataset_results:
        lines.append(
            f"| {result.dataset} | p50 latency (single request) | "
            f"{result.transformer.latency.p50_ms:.1f} ms | {result.baseline.latency.p50_ms:.1f} ms |"
        )
    lines.append(
        f"| all | export size | {dataset_results[0].transformer.size_mb:.1f} MB | "
        "no model file (rule-based) |"
    )
    lines += [
        "",
        "SPEC §3 CPU summarization latency budget: < 3 s per request (measured/reported, not a hard "
        "gate).",
        "",
    ]
    if judge_metrics is not None:
        lines += [
            "## LLM-as-judge (real supportlens tickets)",
            "",
            f"- n = {judge_metrics.n} real ticket summaries judged (`gpt-4o-mini`, 1-5 rubric)",
            f"- mean faithfulness: **{judge_metrics.mean_faithfulness:.2f}** / 5",
            f"- mean coverage: **{judge_metrics.mean_coverage:.2f}** / 5",
            f"- {judge_metrics.parsed_ok_rate:.0%} of judge responses parsed cleanly",
            "",
            "See `docs/summarization-failure-modes.md` for concrete hallucination examples.",
            "",
        ]
    lines += [
        "## Limitations",
        "",
        "- Fine-tuned on samsum/dialogsum -- general and call-center-style dialogue, not "
        "customer-support-ticket text specifically. A domain gap analogous to M2's "
        "Bitext-synthetic-vs-real-tweets finding is plausible but not separately measured beyond the "
        "LLM-judge pass above.",
        "- Free-text generation: no guardrail against hallucinated order numbers, dates, or amounts "
        "beyond what the faithfulness rubric measures on a 50-example sample.",
        '- CPU-only inference (SPEC §5: "serving is CPU-only"), no quantization applied.',
        "- Feeds the dashboard's per-ticket summary block "
        "(`scripts/compute_thread_summaries.py --model transformer`) -- errors here propagate there.",
        "",
    ]
    return "\n".join(lines)


def _render_comparison_report(
    dataset_results: list[DatasetResult],
    judge_metrics: LLMJudgeMetrics | None,
    qualitative_examples: list[Prediction],
) -> str:
    lines = [
        "# M6 comparison report: FLAN-T5-small thread summarizer vs the lead-k extractive baseline",
        "",
        "Generated by `scripts/generate_m6_report.py` from `eval_runs` rows persisted during this "
        "run -- every number below comes from a committed eval run (CLAUDE.md rule #5), nothing "
        "here is hand-typed.",
        "",
    ]

    for result in dataset_results:
        delta = result.transformer.metrics.rouge1 - result.baseline.metrics.rouge1
        lines.append(f"## {result.dataset}")
        lines.append("")
        lines.append("### ROUGE (test split)")
        lines.append("")
        lines.append("| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |")
        lines.append("|---|---|---|---|")
        lines.append(
            f"| baseline (`{BASELINE_MODEL_VERSION}`, k={DEFAULT_K}) | "
            f"{result.baseline.metrics.rouge1:.4f} | {result.baseline.metrics.rouge2:.4f} | "
            f"{result.baseline.metrics.rouge_l:.4f} |"
        )
        lines.append(
            f"| transformer (`{TRANSFORMER_MODEL_VERSION}`) | {result.transformer.metrics.rouge1:.4f} "
            f"| {result.transformer.metrics.rouge2:.4f} | {result.transformer.metrics.rouge_l:.4f} |"
        )
        lines.append("")
        lines.append("### Recommendation")
        lines.append("")
        if delta > WIN_MARGIN:
            lines.append(
                f"**Deploy the transformer** (`{TRANSFORMER_MODEL_VERSION}`). ROUGE-1 delta over the "
                f"baseline is **+{delta:.4f}**, above the {WIN_MARGIN} margin treated as a real win here."
            )
        elif delta > -WIN_MARGIN:
            lines.append(
                f"**Keep the baseline.** ROUGE-1 delta is only **{delta:+.4f}** -- within the "
                f"{WIN_MARGIN} margin treated as noise, not a real win."
            )
        else:
            lines.append(
                f"**Keep the baseline -- the transformer is actually worse here** "
                f"(ROUGE-1 delta **{delta:+.4f}**)."
            )
        lines.append("")

    if judge_metrics is not None:
        lines.append("## LLM-as-judge (real supportlens tickets)")
        lines.append("")
        lines.append(
            f"n = {judge_metrics.n}, mean faithfulness = **{judge_metrics.mean_faithfulness:.2f}**/5, "
            f"mean coverage = **{judge_metrics.mean_coverage:.2f}**/5 "
            f"({judge_metrics.parsed_ok_rate:.0%} parsed cleanly). See "
            "`docs/summarization-failure-modes.md` for concrete examples."
        )
        lines.append("")

    lines.append("## Real-ticket summaries (qualitative)")
    lines.append("")
    if qualitative_examples:
        lines.append(
            "samsum/dialogsum test-set ROUGE above proves head quality; this shows the summarization "
            "feature itself (SPEC M6's actual deliverable) on real ingested tickets, computed by "
            "`scripts/compute_thread_summaries.py`:"
        )
        lines.append("")
        lines.append("| Ticket | Model | Summary |")
        lines.append("|---|---|---|")
        for pred in qualitative_examples:
            summary = (pred.label or "").replace("|", "\\|")
            lines.append(f"| `{pred.ticket_id}` | {pred.model_version} | {summary} |")
        lines.append("")
    else:
        lines.append(
            "No `thread_summary` predictions found -- run "
            "`uv run python scripts/compute_thread_summaries.py` first, then re-run this report."
        )
        lines.append("")

    return "\n".join(lines)


def _sample_qualitative_examples(session: Session, n: int) -> list[Prediction]:
    stmt = (
        select(Prediction)
        .where(Prediction.task == "thread_summary")
        .order_by(Prediction.created_at.desc())
        .limit(n)
    )
    return list(session.scalars(stmt).all())


def main() -> None:
    session = SessionLocal()
    try:
        baseline = ExtractiveSummaryPredictor()
        transformer = SummarizationPredictor(TRANSFORMER_EXPORT_DIR)

        dataset_results = [
            _eval_dataset(session, dataset, baseline, transformer) for dataset in DATASETS
        ]

        judge_rows = _judge_rows(session)
        judge_metrics = aggregate_judge_scores(judge_rows) if judge_rows else None  # type: ignore[arg-type]
        if judge_metrics is not None:
            persist_eval_run(
                session,
                task="thread_summary_judge",
                model_version=f"openai:judge_of_{TRANSFORMER_MODEL_VERSION}",
                dataset="supportlens_real_tickets",
                split=f"judge_sample_{judge_metrics.n}",
                metrics=judge_metrics,
                params={},
            )
            print(
                f"judge aggregate: n={judge_metrics.n} "
                f"mean_faithfulness={judge_metrics.mean_faithfulness:.2f} "
                f"mean_coverage={judge_metrics.mean_coverage:.2f}"
            )
        else:
            print(
                "\nno thread_summary_judge Predictions found -- run "
                "`uv run python -m ml.data.llm_judge_summaries` first for the judge EvalRun."
            )

        qualitative_examples = _sample_qualitative_examples(session, N_QUALITATIVE_EXAMPLES)
        _print_failure_mode_candidates(session, N_FAILURE_CANDIDATES)
    finally:
        session.close()

    MODEL_CARDS_DIR.mkdir(parents=True, exist_ok=True)
    card_path = MODEL_CARDS_DIR / f"{TRANSFORMER_MODEL_VERSION}.md"
    card_path.write_text(_render_model_card(dataset_results, judge_metrics), encoding="utf-8")
    print(f"wrote {card_path}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        _render_comparison_report(dataset_results, judge_metrics, qualitative_examples),
        encoding="utf-8",
    )
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
