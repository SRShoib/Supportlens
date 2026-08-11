"""Evaluates M5's sentiment/emotion baselines and transformer exports on the
real tweet_eval test splits, persists EvalRun rows (CLAUDE.md rule #5: no
metric without an eval run), benchmarks CPU latency + export size, and
renders docs/m5-comparison-report.md + docs/model-cards/*.md from the
results -- nothing in either doc is hand-typed. Structurally the same shape
as scripts/generate_m3_report.py (SPEC M5 is classification, like M3, not
span-level like M4).

Also renders a qualitative section showing real per-ticket trajectories
computed by scripts/compute_sentiment_trajectories.py -- tweet_eval's own
test-set accuracy proves head quality, but the trajectory feature itself
needs a real-ticket demonstration, same instinct as M2/M4's synthetic-vs-real
sections. Skipped gracefully if that script hasn't been run yet.

Run: uv run python scripts/generate_m5_report.py
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from api.db.models import EvalRun, Prediction
from api.db.session import SessionLocal
from sqlalchemy import select
from sqlalchemy.orm import Session

from ml.evaluation.latency import LatencyResult, benchmark_latency
from ml.evaluation.metrics import (
    ClassificationMetrics,
    compute_classification_metrics,
    persist_eval_run,
)
from ml.inference.baseline import BaselinePredictor
from ml.inference.transformer import TransformerPredictor
from ml.training.splits import load_splits

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
CONFIGS_DIR = ROOT / "ml" / "training" / "configs"
MODEL_CARDS_DIR = ROOT / "docs" / "model-cards"
REPORT_PATH = ROOT / "docs" / "m5-comparison-report.md"

TASKS = ["sentiment", "emotion"]
DATASET_BY_TASK = {"sentiment": "tweet_eval_sentiment_v1", "emotion": "tweet_eval_emotion_v1"}
BENCHMARK_TEXT_BY_TASK = {
    "sentiment": "I love this, thank you so much for the quick help!",
    "emotion": "I am so frustrated, this has been broken for a week now.",
}
PREDICT_BATCH_SIZE = 64
WIN_MARGIN = 0.02  # same convention as scripts/generate_m3_report.py
N_TRAJECTORY_EXAMPLES = 5


@dataclass(frozen=True)
class TransformerConfig:
    task: str
    model_slug: str
    hyperparams: dict[str, Any]


@dataclass(frozen=True)
class VariantResult:
    task: str
    model_slug: str
    model_version: str
    export_dir: Path
    metrics: ClassificationMetrics
    latency: LatencyResult
    size_mb: float
    hyperparams: dict[str, Any]


@dataclass(frozen=True)
class BaselineResult:
    task: str
    model_version: str
    metrics: ClassificationMetrics
    latency: LatencyResult
    size_mb: float


def _iter_transformer_configs() -> list[TransformerConfig]:
    configs = []
    for config_path in sorted(CONFIGS_DIR.glob("*.yaml")):
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if raw["task"] not in TASKS:
            continue
        model_slug = raw["model_name"].split("/")[-1]
        configs.append(TransformerConfig(task=raw["task"], model_slug=model_slug, hyperparams=raw))
    return configs


def _dir_size_mb(path: Path) -> float:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 * 1024)


def _predict_in_batches(predictor: TransformerPredictor, texts: list[str]) -> list[str]:
    labels = []
    for start in range(0, len(texts), PREDICT_BATCH_SIZE):
        batch = texts[start : start + PREDICT_BATCH_SIZE]
        labels.extend(r.label for r in predictor.predict(batch))
    return labels


def _eval_transformer_variant(
    session: Session, config: TransformerConfig, test_df: Any, labels: list[str]
) -> VariantResult:
    export_dir = MODELS_DIR / f"transformer_{config.task}_{config.model_slug}_v1" / "final"
    print(f"  loading {export_dir}")
    predictor = TransformerPredictor(export_dir)

    print(f"  predicting {len(test_df)} test rows in batches of {PREDICT_BATCH_SIZE}")
    y_pred = _predict_in_batches(predictor, test_df["text"].tolist())
    metrics = compute_classification_metrics(test_df["label"].tolist(), y_pred, labels)

    model_version = f"transformer_{config.model_slug}_v1"
    persist_eval_run(
        session,
        task=config.task,
        model_version=model_version,
        dataset=DATASET_BY_TASK[config.task],
        split="test",
        metrics=metrics,
        params={"export_dir": str(export_dir), **config.hyperparams},
    )

    latency = benchmark_latency(predictor, BENCHMARK_TEXT_BY_TASK[config.task])
    size_mb = _dir_size_mb(export_dir)
    print(f"  test macro_f1={metrics.macro_f1:.4f} p50={latency.p50_ms:.1f}ms size={size_mb:.1f}MB")

    return VariantResult(
        task=config.task,
        model_slug=config.model_slug,
        model_version=model_version,
        export_dir=export_dir,
        metrics=metrics,
        latency=latency,
        size_mb=size_mb,
        hyperparams=config.hyperparams,
    )


def _selected_baseline_eval_run(session: Session, task: str) -> EvalRun:
    stmt = (
        select(EvalRun)
        .where(EvalRun.task == task, EvalRun.split == "test")
        .order_by(EvalRun.finished_at.desc())
    )
    for run in session.scalars(stmt).all():
        if run.model_version.startswith("baseline_") and run.params.get("selected"):
            return run
    raise RuntimeError(
        f"no selected baseline EvalRun for task={task} — run `make train-baseline-{task}` first"
    )


def _eval_baseline(session: Session, task: str) -> BaselineResult:
    run = _selected_baseline_eval_run(session, task)
    model_path = MODELS_DIR / f"baseline_{task}_v1" / "model.joblib"
    predictor = BaselinePredictor(model_path)
    latency = benchmark_latency(predictor, BENCHMARK_TEXT_BY_TASK[task])
    size_mb = model_path.stat().st_size / (1024 * 1024)
    metrics = ClassificationMetrics(
        macro_f1=run.metrics["macro_f1"],
        per_class_f1=run.metrics["per_class_f1"],
        confusion_matrix=run.metrics["confusion_matrix"],
        labels=run.metrics["labels"],
    )
    return BaselineResult(
        task=task,
        model_version=run.model_version,
        metrics=metrics,
        latency=latency,
        size_mb=size_mb,
    )


def _render_model_card(result: VariantResult, baseline: BaselineResult) -> str:
    hp = result.hyperparams
    lines = [
        f"# Model card: {result.model_version} ({result.task})",
        "",
        f"**Base model:** `{hp['model_name']}`",
        f"**Task:** {result.task} classification, {len(result.metrics.labels)} classes",
        "",
        "## Data & splits",
        "",
        f"- Dataset: `{DATASET_BY_TASK[result.task]}` (HF `tweet_eval`, config `{result.task}`), "
        f"split file `data/splits/{result.task}_v1.parquet`",
        "- Split: tweet_eval's own fixed train/validation/test partition, used verbatim -- not a "
        "fresh seed-42 re-split (see `docs/decisions.md`).",
        f"- Test rows evaluated: {sum(sum(row) for row in result.metrics.confusion_matrix)}",
        "",
        "## Hyperparameters",
        "",
        "| Param | Value |",
        "|---|---|",
    ]
    for key in (
        "num_epochs",
        "batch_size",
        "learning_rate",
        "max_seq_length",
        "warmup_ratio",
        "weight_decay",
    ):
        lines.append(f"| {key} | {hp[key]} |")
    lines += [
        "",
        "## Metrics (test split)",
        "",
        f"- Macro-F1: **{result.metrics.macro_f1:.4f}** (baseline `{baseline.model_version}`: "
        f"{baseline.metrics.macro_f1:.4f})",
        "",
        "Per-class F1:",
        "",
        "| Class | F1 |",
        "|---|---|",
    ]
    for label, f1 in sorted(result.metrics.per_class_f1.items(), key=lambda kv: kv[1]):
        lines.append(f"| {label} | {f1:.4f} |")
    lines += [
        "",
        "## CPU latency & size",
        "",
        "| | Transformer | Baseline |",
        "|---|---|---|",
        f"| p50 latency (single request) | {result.latency.p50_ms:.1f} ms | {baseline.latency.p50_ms:.1f} ms |",
        f"| p95 latency | {result.latency.p95_ms:.1f} ms | {baseline.latency.p95_ms:.1f} ms |",
        f"| Export size | {result.size_mb:.1f} MB | {baseline.size_mb:.1f} MB |",
        "",
        "SPEC §3 CPU classification latency budget: < 150 ms per request (measured/reported, not a "
        "hard gate).",
        "",
        "## Limitations",
        "",
        "- Fine-tuned on tweet_eval -- general Twitter text, not customer-support-specific. Applied "
        "here to support tickets via transfer learning; a domain gap analogous to M2's "
        "Bitext-synthetic-vs-real-tweets finding is plausible but not separately measured here "
        "(no human-labeled sentiment/emotion ground truth exists for this repo's own ticket corpus).",
        f"- Truncates input to {hp['max_seq_length']} tokens; longer messages lose trailing context.",
        '- CPU-only inference (SPEC §5: "serving is CPU-only"), no quantization applied.',
    ]
    if result.task == "sentiment":
        lines.append(
            "- Feeds `ml/inference/sentiment_trajectory.py`'s per-ticket trajectory and "
            "resolution-quality heuristic when `scripts/compute_sentiment_trajectories.py` is run "
            "with `--model transformer` -- errors here propagate into that aggregate."
        )
    lines.append("")
    return "\n".join(lines)


def _render_comparison_report(
    baselines: dict[str, BaselineResult],
    variants: dict[str, list[VariantResult]],
    trajectory_examples: list[Prediction],
) -> str:
    lines = [
        "# M5 comparison report: sentiment/emotion transformer fine-tunes vs classical baselines",
        "",
        "Generated by `scripts/generate_m5_report.py` from `eval_runs` rows persisted during this "
        "run -- every number below comes from a committed eval run (CLAUDE.md rule #5), nothing "
        "here is hand-typed.",
        "",
    ]

    for task in TASKS:
        baseline = baselines[task]
        task_variants = sorted(variants[task], key=lambda v: v.metrics.macro_f1, reverse=True)
        winner = task_variants[0]
        delta = winner.metrics.macro_f1 - baseline.metrics.macro_f1

        lines.append(f"## {task.capitalize()}")
        lines.append("")
        lines.append("### Accuracy (test macro-F1)")
        lines.append("")
        lines.append("| Model | Macro-F1 |")
        lines.append("|---|---|")
        lines.append(f"| baseline (`{baseline.model_version}`) | {baseline.metrics.macro_f1:.4f} |")
        for variant in task_variants:
            lines.append(
                f"| transformer (`{variant.model_version}`) | {variant.metrics.macro_f1:.4f} |"
            )
        lines.append("")

        lines.append("### Latency (CPU, single request, p50/p95)")
        lines.append("")
        lines.append("| Model | p50 ms | p95 ms |")
        lines.append("|---|---|---|")
        lines.append(
            f"| baseline | {baseline.latency.p50_ms:.1f} | {baseline.latency.p95_ms:.1f} |"
        )
        for variant in task_variants:
            lines.append(
                f"| {variant.model_slug} | {variant.latency.p50_ms:.1f} | {variant.latency.p95_ms:.1f} |"
            )
        lines.append("")

        lines.append("### Export size")
        lines.append("")
        lines.append("| Model | Size |")
        lines.append("|---|---|")
        lines.append(f"| baseline | {baseline.size_mb:.1f} MB |")
        for variant in task_variants:
            lines.append(f"| {variant.model_slug} | {variant.size_mb:.1f} MB |")
        lines.append("")

        lines.append("### Recommendation")
        lines.append("")
        if delta > WIN_MARGIN:
            lines.append(
                f"**Deploy the transformer** (`{winner.model_version}`). Macro-F1 delta over the "
                f"baseline is **+{delta:.4f}**, above the {WIN_MARGIN} margin treated as a real win "
                f"here — worth the {winner.size_mb / baseline.size_mb:.0f}x size and "
                f"{winner.latency.p50_ms / baseline.latency.p50_ms:.1f}x latency cost."
            )
        elif delta > -WIN_MARGIN:
            lines.append(
                f"**Keep the baseline.** Macro-F1 delta is only **{delta:+.4f}** — within the "
                f"{WIN_MARGIN} margin treated as noise, not a real win. The transformer "
                f"(`{winner.model_version}`) costs {winner.size_mb / baseline.size_mb:.0f}x the "
                "size and more latency for accuracy that isn't meaningfully better."
            )
        else:
            lines.append(
                f"**Keep the baseline — the transformer is actually worse here** "
                f"(macro-F1 delta **{delta:+.4f}**)."
            )
        lines.append("")

    lines.append("## Real-ticket trajectories (qualitative)")
    lines.append("")
    if trajectory_examples:
        lines.append(
            "tweet_eval's test-set accuracy above proves head quality; this shows the trajectory "
            "feature itself (SPEC M5's actual deliverable) on real ingested tickets, computed by "
            "`scripts/compute_sentiment_trajectories.py`:"
        )
        lines.append("")
        lines.append("| Ticket | Sequence | Ending | Resolution quality |")
        lines.append("|---|---|---|---|")
        for pred in trajectory_examples:
            sequence = " → ".join(pred.payload.get("sequence", []))
            lines.append(f"| `{pred.ticket_id}` | {sequence} | {pred.label} | {pred.score:+.3f} |")
        lines.append("")
    else:
        lines.append(
            "No `sentiment_trajectory` predictions found — run "
            "`uv run python scripts/compute_sentiment_trajectories.py` first, then re-run this report."
        )
        lines.append("")

    return "\n".join(lines)


def _sample_trajectory_examples(session: Session, n: int) -> list[Prediction]:
    stmt = (
        select(Prediction)
        .where(Prediction.task == "sentiment_trajectory")
        .order_by(Prediction.created_at.desc())
        .limit(n)
    )
    return list(session.scalars(stmt).all())


def main() -> None:
    session = SessionLocal()
    try:
        transformer_configs = _iter_transformer_configs()

        baselines: dict[str, BaselineResult] = {}
        variants: dict[str, list[VariantResult]] = {task: [] for task in TASKS}

        for task in TASKS:
            print(f"task={task}: baseline benchmark")
            baselines[task] = _eval_baseline(session, task)

            df = load_splits(f"{task}_v1")
            labels = sorted(df["label"].unique())
            test_df = df[df["split"] == "test"]

            for config in [c for c in transformer_configs if c.task == task]:
                print(f"task={task}: evaluating {config.model_slug}")
                variants[task].append(_eval_transformer_variant(session, config, test_df, labels))

        trajectory_examples = _sample_trajectory_examples(session, N_TRAJECTORY_EXAMPLES)
    finally:
        session.close()

    if any(not variants[task] for task in TASKS):
        raise RuntimeError(
            "no transformer exports found for one or more tasks under "
            "models/transformer_{task}_*/final -- run the GPU training first "
            "(docs/m5-how-to-run-locally.md)"
        )

    MODEL_CARDS_DIR.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        for variant in variants[task]:
            card_path = MODEL_CARDS_DIR / f"{variant.model_version}_{task}.md"
            card_path.write_text(_render_model_card(variant, baselines[task]), encoding="utf-8")
            print(f"wrote {card_path}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        _render_comparison_report(baselines, variants, trajectory_examples), encoding="utf-8"
    )
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
