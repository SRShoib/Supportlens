"""Evaluates M7's two topic-discovery variants (TF-IDF/KMeans baseline vs
BERTopic) on NPMI coherence, persists EvalRun rows (CLAUDE.md rule #5: no
metric without an eval run), and renders docs/m7-comparison-report.md +
docs/model-cards/topics_bertopic_v1.md from the results -- nothing in
either doc is hand-typed.

Unlike M2-M6's supervised comparisons (which pick a winner by held-out
accuracy/ROUGE), M7 is unsupervised: both variants are fit on and scored
against the SAME corpus, no train/test split. Coherence
(ml/evaluation/topic_metrics.py's NPMI, computed here from the actual
corpus documents) stands in for accuracy; the win-margin recommendation
logic mirrors generate_m3_report.py's pattern even though the underlying
metric is different (see docs/decisions.md for why NPMI, not a hardcoded
"coherent" threshold).

Reads exactly what ml/training/topic_model.py exported --
models/topics_{variant}_v1/{topics.json,assignments.parquet} -- plus the
shared data/embeddings/tickets_minilm_v1.parquet documents. Needs neither
the `topics` dependency group nor a real BERTopic/sentence-transformers run
to execute (any topics.json/assignments.parquet/documents.parquet with the
right shape works, only pandas/pyarrow -- both default deps), but obviously
needs a real run to report real numbers.

Run: uv run python scripts/generate_m7_report.py
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from api.db.session import SessionLocal
from sqlalchemy.orm import Session

from ml.evaluation.metrics import persist_eval_run
from ml.evaluation.topic_metrics import TOP_N_TERMS, CoherenceMetrics, compute_topic_coherence

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
DOCUMENTS_PATH = ROOT / "data" / "embeddings" / "tickets_minilm_v1.parquet"
MODEL_CARDS_DIR = ROOT / "docs" / "model-cards"
REPORT_PATH = ROOT / "docs" / "m7-comparison-report.md"

VARIANTS = ("kmeans", "bertopic")
DEPLOYED_VARIANT = "bertopic"  # matches scripts/assign_topics.py's DEFAULT_VARIANT
DATASET = "twitter_slice_v1"
OUTLIER_TOPIC_KEY = -1
WIN_MARGIN = 0.02  # same convention as generate_m3/m5_report.py, applied to mean NPMI here
N_EXAMPLE_TOPICS = 8
SPEC_MIN_TOPICS = 30


def _tokenize(text: str) -> list[str]:
    return [t for t in "".join(c if c.isalnum() else " " for c in text.lower()).split() if t]


@dataclass(frozen=True)
class VariantResult:
    variant: str
    model_version: str
    n_topics: int  # excludes the outlier cluster
    n_outliers: int
    coherence: CoherenceMetrics
    topics: list[dict[str, Any]]  # raw topics.json["topics"], sorted by size desc


def _load_variant(variant: str, documents: list[list[str]]) -> VariantResult | None:
    topics_path = MODELS_DIR / f"topics_{variant}_v1" / "topics.json"
    if not topics_path.exists():
        return None

    raw = json.loads(topics_path.read_text(encoding="utf-8"))
    topics = sorted(raw["topics"], key=lambda t: t["size"], reverse=True)
    n_outliers = sum(t["size"] for t in topics if t["topic_key"] == OUTLIER_TOPIC_KEY)
    n_topics = sum(1 for t in topics if t["topic_key"] != OUTLIER_TOPIC_KEY)

    topic_terms = {
        t["topic_key"]: t["keywords"] for t in topics if t["topic_key"] != OUTLIER_TOPIC_KEY
    }
    coherence = compute_topic_coherence(topic_terms, documents)

    return VariantResult(
        variant=variant,
        model_version=raw["model_version"],
        n_topics=n_topics,
        n_outliers=n_outliers,
        coherence=coherence,
        topics=topics,
    )


def _persist(session: Session, result: VariantResult) -> None:
    persist_eval_run(
        session,
        task="topics",
        model_version=result.model_version,
        dataset=DATASET,
        split="full",
        metrics=result.coherence,
        params={"n_topics": result.n_topics, "n_outliers": result.n_outliers},
    )


def _top_topics_table(topics: list[dict[str, Any]]) -> list[str]:
    lines = ["| Topic | Size | Keywords |", "|---|---|---|"]
    shown = [t for t in topics if t["topic_key"] != OUTLIER_TOPIC_KEY][:N_EXAMPLE_TOPICS]
    for topic in shown:
        keywords = ", ".join(topic["keywords"][:6])
        lines.append(f"| {topic['label']} | {topic['size']} | {keywords} |")
    return lines


def _render_model_card(result: VariantResult) -> str:
    lines = [
        f"# Model card: {result.model_version}",
        "",
        "**Task:** unsupervised topic discovery (SPEC M7) -- sentence-transformers "
        "embeddings (`sentence-transformers/all-MiniLM-L6-v2`) -> UMAP -> HDBSCAN -> "
        "c-TF-IDF labels, via `ml/training/topic_model.py`.",
        "",
        "## Data",
        "",
        "- Corpus: `TicketSource.TWITTER` tickets only (the real-ticket corpus, SPEC §2) -- "
        "Bitext is excluded (synthetic, `created_at` always NULL, see `docs/decisions.md`).",
        "- Document unit: the concatenation of each ticket's CUSTOMER messages only "
        "(`scripts/compute_embeddings.py`), not the full agent+customer thread.",
        "",
        "## Results",
        "",
        f"- Topics discovered: **{result.n_topics}** (SPEC M7 accept: ≥ {SPEC_MIN_TOPICS}) "
        f"+ {result.n_outliers} tickets in the HDBSCAN outlier cluster.",
        f"- Mean NPMI coherence: **{result.coherence.mean_npmi:.4f}** "
        f"(`ml/evaluation/topic_metrics.py`, top-{TOP_N_TERMS} c-TF-IDF terms per topic).",
        "",
        "## Top topics",
        "",
        *_top_topics_table(result.topics),
        "",
        "## Limitations",
        "",
        "- Unsupervised: no held-out test set, no ground-truth topic labels exist for this "
        "corpus -- coherence (NPMI over the model's own corpus) is a proxy for quality, not "
        "an accuracy number.",
        "- Twitter-only: never fit or evaluated on Bitext's synthetic utterances.",
        "- Topic count and the outlier cluster's size are sensitive to "
        "`ml/training/configs/topics_minilm_bertopic.yaml`'s UMAP/HDBSCAN hyperparameters -- "
        "not re-tuned per corpus slice.",
        "- Labels shown above are c-TF-IDF keyword joins, the committed default -- SPEC M7's "
        "optional LLM naming pass (`ml/data/llm_topic_labels.py`, default off) may have "
        "since overwritten some of them; this card doesn't track that provenance.",
        "",
    ]
    return "\n".join(lines)


def _render_recommendation(baseline: VariantResult, bertopic: VariantResult) -> list[str]:
    delta = bertopic.coherence.mean_npmi - baseline.coherence.mean_npmi
    lines = ["### Recommendation", ""]
    if delta > WIN_MARGIN:
        lines.append(
            f"**Deploy BERTopic** (`{bertopic.model_version}`). Mean NPMI delta over the "
            f"KMeans baseline is **+{delta:.4f}**, above the {WIN_MARGIN} margin treated as a "
            "real win here -- and SPEC M7 explicitly names BERTopic's density clustering "
            "(UMAP + HDBSCAN) as the deliverable, which the KMeans baseline structurally "
            f"can't reproduce: it always produces exactly `kmeans_n_clusters` topics with no "
            "outlier cluster, regardless of whether the corpus actually clusters that way."
        )
    elif delta > -WIN_MARGIN:
        lines.append(
            f"**Coherence is a wash** (delta **{delta:+.4f}**, within the {WIN_MARGIN} margin "
            f"treated as noise). BERTopic (`{bertopic.model_version}`) is still the deployed "
            "variant (`scripts/assign_topics.py`'s default): SPEC M7 explicitly asks for "
            "density clustering plus the outlier cluster and trend-detection story, which "
            "the fixed-k KMeans baseline can't represent."
        )
    else:
        lines.append(
            f"**KMeans scores higher on coherence** (delta **{delta:+.4f}**), but BERTopic "
            f"(`{bertopic.model_version}`) remains the deployed variant "
            "(`scripts/assign_topics.py`'s default) -- SPEC M7 explicitly names it as the "
            "pipeline, and only BERTopic produces the outlier cluster and variable topic "
            "count the emerging-issues trend detection is built around."
        )
    lines.append("")
    return lines


def _render_comparison_report(results: dict[str, VariantResult]) -> str:
    lines = [
        "# M7 comparison report: BERTopic vs TF-IDF/KMeans baseline",
        "",
        "Generated by `scripts/generate_m7_report.py` from `eval_runs` rows persisted during "
        "this run -- every number below comes from a committed eval run (CLAUDE.md rule #5), "
        "nothing here is hand-typed.",
        "",
        'CLAUDE.md ground rule #2 ("baselines before transformers, always -- the comparison '
        "IS the deliverable\") applies here even though SPEC M7's text names only BERTopic "
        "-- see `docs/decisions.md`.",
        "",
        "## Topic count & coherence",
        "",
        "| Variant | Topics (excl. outliers) | Outlier tickets | Mean NPMI |",
        "|---|---|---|---|",
    ]
    for variant in VARIANTS:
        result = results.get(variant)
        if result is None:
            lines.append(f"| {variant} | — | — | (not run) |")
            continue
        lines.append(
            f"| {variant} (`{result.model_version}`) | {result.n_topics} | "
            f"{result.n_outliers} | {result.coherence.mean_npmi:.4f} |"
        )
    lines.append("")

    if results.get("kmeans") is not None and results.get("bertopic") is not None:
        lines += _render_recommendation(results["kmeans"], results["bertopic"])

    lines.append(f"## SPEC M7 acceptance: ≥ {SPEC_MIN_TOPICS} coherent topics")
    lines.append("")
    deployed = results.get(DEPLOYED_VARIANT)
    if deployed is not None:
        verdict = "PASS" if deployed.n_topics >= SPEC_MIN_TOPICS else "FAIL"
        lines.append(
            f"`{deployed.model_version}` discovered **{deployed.n_topics}** topics (excluding "
            f"the {deployed.n_outliers}-ticket outlier cluster) -- **{verdict}** against the "
            f"≥ {SPEC_MIN_TOPICS} bar."
        )
    else:
        lines.append(f"`{DEPLOYED_VARIANT}` has not been run yet.")
    lines.append("")

    for variant in VARIANTS:
        result = results.get(variant)
        if result is None:
            continue
        lines.append(f"## {variant} -- top topics")
        lines.append("")
        lines += _top_topics_table(result.topics)
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    if not DOCUMENTS_PATH.exists():
        raise RuntimeError(
            f"{DOCUMENTS_PATH} not found -- run `make embed-tickets` first "
            "(docs/m7-how-to-run-locally.md)"
        )
    documents = [_tokenize(doc) for doc in pd.read_parquet(DOCUMENTS_PATH)["document"]]

    results: dict[str, VariantResult] = {}
    for variant in VARIANTS:
        result = _load_variant(variant, documents)
        if result is None:
            print(f"skipping {variant}: models/topics_{variant}_v1/topics.json not found")
            continue
        results[variant] = result
        print(
            f"{variant}: {result.n_topics} topics (+{result.n_outliers} outliers), "
            f"mean NPMI={result.coherence.mean_npmi:.4f}"
        )

    if not results:
        raise RuntimeError(
            "no topic model exports found under models/topics_*_v1/ -- run `make fit-topics` "
            "first (docs/m7-how-to-run-locally.md)"
        )

    session = SessionLocal()
    try:
        for result in results.values():
            _persist(session, result)
    finally:
        session.close()

    if DEPLOYED_VARIANT in results:
        MODEL_CARDS_DIR.mkdir(parents=True, exist_ok=True)
        card_path = MODEL_CARDS_DIR / f"{results[DEPLOYED_VARIANT].model_version}.md"
        card_path.write_text(_render_model_card(results[DEPLOYED_VARIANT]), encoding="utf-8")
        print(f"wrote {card_path}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(_render_comparison_report(results), encoding="utf-8")
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
