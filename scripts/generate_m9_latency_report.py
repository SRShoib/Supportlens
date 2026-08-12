"""Benchmarks CPU single-request latency for every servable predictor
across M2-M7 and persists each as an EvalRun (SPEC M9: "/metrics area...
latency percentiles"; accept criterion: "all metrics render from Postgres
eval runs"). Every M3-M6 report script already computes a LatencyResult
via ml/evaluation/latency.py's benchmark_latency() but never persists it
(only prints it / writes it into that script's own markdown report and
model cards) -- this script is the first to persist any of them.

Deliberately its own script rather than folded into M3-M6's report
scripts: those scripts' latency benchmarks are a side effect of
re-running a full test-set evaluation (thousands of predictions,
sometimes GPU-fine-tuned transformer inference over a full split) --
reusing that code path would mean re-running expensive evaluation just
to add a latency number, and would duplicate accuracy EvalRun rows on
every latency-only rerun (that table has no upsert, see
docs/decisions.md's M6/M7 duplicate-row entries). This script only
loads each already-exported model once and times a fixed probe text --
`dataset="latency_probe"` on every row here signals that, distinct from
the real dataset names accuracy EvalRuns use for the same task.

Skips gracefully (prints, doesn't raise) whenever a model export isn't
present locally -- same "SKIPPED" convention scripts/generate_m4_report.py
already uses for missing transformer exports.

model_version strings deliberately match whatever each task's own
accuracy EvalRun already uses (scripts/generate_m3/m4/m5/m6_report.py),
so a latency row and its accuracy counterpart group together under the
same (task, model_version) in GET /eval-runs -- except "hybrid_ner_v1"
(the rules+model router apps/api/routers/predict.py actually serves for
model="transformer") and "all-MiniLM-L6-v2" (M7's embedder, which has no
accuracy EvalRun at all -- topic coherence is scored on the topic model,
not the embedder itself), both new identifiers introduced here.

Run: uv run python scripts/generate_m9_latency_report.py
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from api.db.session import SessionLocal
from sqlalchemy.orm import Session

from ml.evaluation.latency import LatencyResult, benchmark_latency
from ml.evaluation.metrics import persist_eval_run
from ml.inference.base import Predictor
from ml.inference.baseline import BaselinePredictor
from ml.inference.extractive_summary import ExtractiveSummaryPredictor
from ml.inference.hybrid_ner import HybridEntityPredictor
from ml.inference.rules_ner import RulesEntityPredictor
from ml.inference.summarization import SummarizationPredictor
from ml.inference.token_classification import TokenClassificationPredictor
from ml.inference.transformer import TransformerPredictor

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
REPORT_PATH = ROOT / "docs" / "m9-latency-report.md"

DATASET = "latency_probe"

INTENT_TEXT = "I want to cancel my order and get a refund, please help me now."
URGENCY_TEXT = "This is unacceptable, I've been waiting 3 days and no one has responded!"
SENTIMENT_TEXT = "I love this, thank you so much for the quick help!"
EMOTION_TEXT = "I am so frustrated, this has been broken for a week now."
ENTITY_TEXT = "order ORD-99321 shipped yesterday, charged $49.99 for my iPhone 12 Pro Max"
DIALOGUE_TEXT = (
    "Customer: my order has not arrived and it was supposed to be here 3 days ago\n"
    "Agent: I am sorry, let me check the tracking\n"
    "Agent: it looks like it is stuck at the depot, I will escalate it\n"
    "Customer: thank you please hurry"
)
EMBEDDING_TEXT = "my package never arrived, where is it, this is the third time this has happened"

# SPEC §3's CPU latency budgets, per request -- measured/reported, not a
# hard gate (same framing every M3-M6 report already uses).
BUDGET_MS_BY_TASK = {
    "intent": 150.0,
    "urgency": 150.0,
    "entities": 250.0,
    "sentiment": 150.0,
    "emotion": 150.0,
    "thread_summary": 3000.0,
    "embedding": 100.0,
}


@dataclass(frozen=True)
class LatencyTarget:
    task: str
    model_version: str
    text: str
    build: Callable[[], Predictor]
    export_path: Path | None  # None => always available (rules/extractive baselines)


def _baseline(task: str, model_version: str, text: str, model_dir_name: str) -> LatencyTarget:
    path = MODELS_DIR / model_dir_name / "model.joblib"
    return LatencyTarget(task, model_version, text, lambda: BaselinePredictor(path), path)


def _transformer(task: str, model_version: str, text: str, export_dir_name: str) -> LatencyTarget:
    export_dir = MODELS_DIR / export_dir_name / "final"
    return LatencyTarget(
        task, model_version, text, lambda: TransformerPredictor(export_dir), export_dir
    )


def _token_classification(
    task: str, model_version: str, text: str, export_dir_name: str
) -> LatencyTarget:
    """Same shape as _transformer above, but for M4's token-classification
    exports specifically -- loading those through TransformerPredictor's
    AutoModelForSequenceClassification would attach the wrong head (a
    freshly-initialized classification pooler on top of a token-classification
    checkpoint), which is exactly the "not initialized from the checkpoint"
    warning that surfaced when this script's first draft reused _transformer
    for entities. Timing the wrong architecture would also be a real
    inaccuracy, not just a cosmetic one."""
    export_dir = MODELS_DIR / export_dir_name / "final"
    return LatencyTarget(
        task, model_version, text, lambda: TokenClassificationPredictor(export_dir), export_dir
    )


def _build_hybrid_entity_predictor() -> Predictor:
    routing_path = MODELS_DIR / "entity_routing_v1.json"
    routing_config = json.loads(routing_path.read_text(encoding="utf-8"))
    model_predictor = TokenClassificationPredictor(ROOT / routing_config["model_export_dir"])
    return HybridEntityPredictor(RulesEntityPredictor(), model_predictor, routing_config["labels"])


def _build_targets() -> list[LatencyTarget]:
    return [
        _baseline("intent", "baseline_intent_v1", INTENT_TEXT, "baseline_intent_v1"),
        _transformer(
            "intent",
            "transformer_distilbert-base-uncased_v1",
            INTENT_TEXT,
            "transformer_intent_distilbert-base-uncased_v1",
        ),
        _transformer(
            "intent",
            "transformer_deberta-v3-small_v1",
            INTENT_TEXT,
            "transformer_intent_deberta-v3-small_v1",
        ),
        _baseline("urgency", "baseline_urgency_v1", URGENCY_TEXT, "baseline_urgency_v1"),
        _transformer(
            "urgency",
            "transformer_distilbert-base-uncased_v1",
            URGENCY_TEXT,
            "transformer_urgency_distilbert-base-uncased_v1",
        ),
        _transformer(
            "urgency",
            "transformer_deberta-v3-small_v1",
            URGENCY_TEXT,
            "transformer_urgency_deberta-v3-small_v1",
        ),
        _baseline("sentiment", "baseline_sentiment_v1", SENTIMENT_TEXT, "baseline_sentiment_v1"),
        _transformer(
            "sentiment",
            "transformer_distilbert-base-uncased_v1",
            SENTIMENT_TEXT,
            "transformer_sentiment_distilbert-base-uncased_v1",
        ),
        _baseline("emotion", "baseline_emotion_v1", EMOTION_TEXT, "baseline_emotion_v1"),
        _transformer(
            "emotion",
            "transformer_distilbert-base-uncased_v1",
            EMOTION_TEXT,
            "transformer_emotion_distilbert-base-uncased_v1",
        ),
        LatencyTarget("entities", "rules_ner_v1", ENTITY_TEXT, RulesEntityPredictor, None),
        _token_classification(
            "entities",
            "transformer_entities_bert-base-cased_v1",
            ENTITY_TEXT,
            "transformer_entities_bert-base-cased_v1",
        ),
        _token_classification(
            "entities",
            "transformer_entities_distilbert-base-cased_v1",
            ENTITY_TEXT,
            "transformer_entities_distilbert-base-cased_v1",
        ),
        LatencyTarget(
            "entities",
            "hybrid_ner_v1",
            ENTITY_TEXT,
            _build_hybrid_entity_predictor,
            MODELS_DIR / "entity_routing_v1.json",
        ),
        LatencyTarget(
            "thread_summary",
            "baseline_thread_summary_v1",
            DIALOGUE_TEXT,
            ExtractiveSummaryPredictor,
            None,
        ),
        LatencyTarget(
            "thread_summary",
            "transformer_thread_summary_flan-t5-small_v1",
            DIALOGUE_TEXT,
            lambda: SummarizationPredictor(
                MODELS_DIR / "transformer_thread_summary_flan-t5-small_v1" / "final"
            ),
            MODELS_DIR / "transformer_thread_summary_flan-t5-small_v1" / "final",
        ),
    ]


def _dir_size_mb(path: Path) -> float | None:
    # hybrid_ner_v1's export_path points at entity_routing_v1.json (used
    # only as the existence gate -- the hybrid predictor has no export of
    # its own, it loads the routing file's referenced model directory) --
    # that file's own size (a few hundred bytes) isn't a meaningful model
    # size, so report n/a rather than a misleading "0.0 MB".
    if path.suffix == ".json":
        return None
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024 * 1024)


@dataclass(frozen=True)
class BenchmarkRow:
    task: str
    model_version: str
    latency: LatencyResult
    size_mb: float | None
    budget_ms: float


def _run_target(session: Session, target: LatencyTarget) -> BenchmarkRow | None:
    if target.export_path is not None and not target.export_path.exists():
        print(f"SKIPPED {target.task}/{target.model_version} -- no export at {target.export_path}")
        return None

    print(f"benchmarking {target.task}/{target.model_version}...")
    predictor = target.build()
    latency = benchmark_latency(predictor, target.text)
    size_mb = _dir_size_mb(target.export_path) if target.export_path is not None else None

    persist_eval_run(
        session,
        task=target.task,
        model_version=target.model_version,
        dataset=DATASET,
        split="latency",
        metrics=latency,
        params={"probe_text_chars": len(target.text)},
    )

    budget = BUDGET_MS_BY_TASK[target.task]
    status = "OK" if latency.p50_ms < budget else "OVER BUDGET"
    print(
        f"  p50={latency.p50_ms:.1f}ms p95={latency.p95_ms:.1f}ms ({status}, budget {budget:.0f}ms)"
    )
    return BenchmarkRow(target.task, target.model_version, latency, size_mb, budget)


def _run_embedding_target(session: Session) -> BenchmarkRow | None:
    # Lazy: sentence-transformers lives behind the `topics`/`search`
    # dependency groups, not `training` -- every other predictor above
    # this line is importable with just `training` installed, so this is
    # the one target that can be legitimately absent on a training-only
    # machine (docs/decisions.md).
    try:
        from ml.inference.embeddings import SentenceEmbeddingPredictor
    except ImportError:
        print(
            "SKIPPED embedding/all-MiniLM-L6-v2 -- sentence-transformers not installed "
            "(uv sync --group topics or --group search)"
        )
        return None

    print("benchmarking embedding/all-MiniLM-L6-v2...")
    predictor = SentenceEmbeddingPredictor()
    latency = benchmark_latency(predictor, EMBEDDING_TEXT)

    persist_eval_run(
        session,
        task="embedding",
        model_version="all-MiniLM-L6-v2",
        dataset=DATASET,
        split="latency",
        metrics=latency,
        params={"probe_text_chars": len(EMBEDDING_TEXT)},
    )

    budget = BUDGET_MS_BY_TASK["embedding"]
    status = "OK" if latency.p50_ms < budget else "OVER BUDGET"
    print(
        f"  p50={latency.p50_ms:.1f}ms p95={latency.p95_ms:.1f}ms ({status}, budget {budget:.0f}ms)"
    )
    return BenchmarkRow("embedding", "all-MiniLM-L6-v2", latency, None, budget)


def _render_report(rows: list[BenchmarkRow]) -> str:
    lines = [
        "# M9 latency report: CPU single-request latency, every servable predictor",
        "",
        "Generated by `scripts/generate_m9_latency_report.py` from `eval_runs` rows persisted "
        'during this run (`split="latency"`, alongside each task\'s own accuracy eval runs) -- '
        "every number below comes from a committed eval run (CLAUDE.md rule #5), nothing here "
        "is hand-typed. Probe text is fixed per task, 20 warm runs after 3 discarded warmup "
        "calls (`ml/evaluation/latency.py::benchmark_latency`), single request at a time -- "
        "SPEC §3's budgets are per-request, not batch throughput.",
        "",
        "| Task | Model | p50 ms | p95 ms | Max ms | Size | Budget | Status |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        size = f"{row.size_mb:.1f} MB" if row.size_mb is not None else "n/a"
        status = "OK" if row.latency.p50_ms < row.budget_ms else "OVER"
        lines.append(
            f"| {row.task} | `{row.model_version}` | {row.latency.p50_ms:.1f} | "
            f"{row.latency.p95_ms:.1f} | {row.latency.max_ms:.1f} | {size} | "
            f"{row.budget_ms:.0f} ms | {status} |"
        )
    lines.append("")

    over_budget = [r for r in rows if r.latency.p50_ms >= r.budget_ms]
    if over_budget:
        lines.append(
            f"**{len(over_budget)} model(s) exceed their SPEC §3 p50 budget** on this machine: "
            + ", ".join(f"`{r.model_version}` ({r.task})" for r in over_budget)
            + ". Budgets are measured/reported per SPEC §3, not hard gates."
        )
    else:
        lines.append("Every benchmarked model is under its SPEC §3 p50 latency budget.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    session = SessionLocal()
    rows: list[BenchmarkRow] = []
    try:
        for target in _build_targets():
            row = _run_target(session, target)
            if row is not None:
                rows.append(row)
        embedding_row = _run_embedding_target(session)
        if embedding_row is not None:
            rows.append(embedding_row)
    finally:
        session.close()

    if not rows:
        raise RuntimeError("no models found to benchmark -- train/export at least one first")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(_render_report(rows), encoding="utf-8")
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
