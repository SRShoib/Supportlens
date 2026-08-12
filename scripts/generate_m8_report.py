"""Evaluates M8's retrieval pipeline -- dense-only vs dense+cross-encoder
rerank -- on the 100-query synthetic eval set
(ml/data/retrieval_eval_set.py), persists EvalRun rows (CLAUDE.md rule #5:
no metric without an eval run), and renders docs/m8-comparison-report.md
from the results -- nothing in it is hand-typed.

Unlike M7's report script, there's no flat-file shortcut here: hit-rate@k
requires actually running retrieval for each of the 100 queries against a
real Chroma index, not reading back something already computed offline.
Needs scripts/index_search_corpus.py to have already run, and the `search`
dependency group installed (sentence-transformers for the query embedder
and cross-encoder, chromadb for the index).

Run: uv run python scripts/generate_m8_report.py
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from api.db.session import SessionLocal
from sqlalchemy.orm import Session

from ml.evaluation.metrics import persist_eval_run
from ml.evaluation.retrieval_metrics import RetrievalMetrics, compute_hit_rate, hit_at_k
from ml.inference.rag_reply import MIN_CONFIDENCE
from ml.inference.reranker import Reranker
from ml.inference.retrieval import retrieve
from ml.inference.vector_store import ChromaVectorStore

ROOT = Path(__file__).resolve().parents[1]
EVAL_SET_PATH = ROOT / "data" / "eval" / "retrieval_queries.parquet"
REPORT_PATH = ROOT / "docs" / "m8-comparison-report.md"

DATASET = "retrieval_eval_v1"
K = 5
# Retrieve more than K per query so hit-rate@5 has real room to differ
# between variants (a pool capped at exactly K would trivially cap both
# variants' hit-rate at the same ceiling).
POOL_SIZE = 20
WIN_MARGIN = 0.02
N_EXAMPLE_QUERIES = 8

# SPEC M8: "RAG endpoint refuses gracefully when retrieval confidence is
# low (no-answer behavior demonstrated)". Run through the real retrieval
# pipeline every time this report regenerates -- not a one-off manual
# check -- so this evidence can't silently go stale (docs/decisions.md has
# the original 5-vs-5 measurement that set MIN_CONFIDENCE).
NO_ANSWER_EXAMPLES: list[tuple[str, str]] = [
    ("on-topic", "my package never arrived, where is it"),
    ("on-topic", "how do I reset my password"),
    ("on-topic", "flight got cancelled and I need a refund"),
    ("off-topic", "what is the capital of France"),
    ("off-topic", "purple elephants dance under the moonlight"),
    ("off-topic", "can you write me a poem about the ocean"),
]


@dataclass(frozen=True)
class VariantResult:
    variant: str
    model_version: str
    metrics: RetrievalMetrics
    per_query_ranked_ids: list[list[str]]


@dataclass(frozen=True)
class NoAnswerExample:
    kind: str  # "on-topic" | "off-topic"
    query: str
    best_score: float
    would_refuse: bool


def run_no_answer_examples(
    examples: list[tuple[str, str]],
    *,
    embedder,
    store: ChromaVectorStore,
    reranker: Reranker,
) -> list[NoAnswerExample]:
    rows = []
    for kind, query in examples:
        ranked = retrieve(
            query, embedder=embedder, store=store, reranker=reranker, pool_size=POOL_SIZE
        )
        best_score = ranked[0].score if ranked else float("-inf")
        rows.append(
            NoAnswerExample(
                kind=kind,
                query=query,
                best_score=best_score,
                would_refuse=best_score < MIN_CONFIDENCE,
            )
        )
    return rows


def run_variant(
    queries: list[str],
    *,
    embedder,
    store: ChromaVectorStore,
    reranker: Reranker | None,
) -> list[list[str]]:
    results = []
    for query in queries:
        ranked = retrieve(
            query, embedder=embedder, store=store, reranker=reranker, pool_size=POOL_SIZE
        )
        results.append([r.hit.id for r in ranked[:K]])
    return results


def _persist(session: Session, result: VariantResult) -> None:
    persist_eval_run(
        session,
        task="retrieval",
        model_version=result.model_version,
        dataset=DATASET,
        split="full",
        metrics=result.metrics,
        params={"k": K, "pool_size": POOL_SIZE, "n_queries": result.metrics.n_queries},
    )


def _example_rows(
    queries: list[str], relevant_ids: list[str], dense: VariantResult, rerank: VariantResult
) -> list[str]:
    lines = ["| Query | Dense hit? | Dense+rerank hit? |", "|---|---|---|"]
    for query, relevant, dense_ids, rerank_ids in list(
        zip(
            queries,
            relevant_ids,
            dense.per_query_ranked_ids,
            rerank.per_query_ranked_ids,
            strict=True,
        )
    )[:N_EXAMPLE_QUERIES]:
        dense_hit = "yes" if hit_at_k(dense_ids, relevant, K) else "no"
        rerank_hit = "yes" if hit_at_k(rerank_ids, relevant, K) else "no"
        snippet = query if len(query) <= 80 else query[:77] + "..."
        lines.append(f"| {snippet} | {dense_hit} | {rerank_hit} |")
    return lines


def _render_recommendation(dense: VariantResult, rerank: VariantResult) -> list[str]:
    delta = rerank.metrics.hit_rate_at_k - dense.metrics.hit_rate_at_k
    lines = ["### Rerank vs no-rerank", ""]
    if delta > WIN_MARGIN:
        lines.append(
            f"**Reranking helps**: hit-rate@{K} rises from **{dense.metrics.hit_rate_at_k:.3f}** "
            f"(dense only) to **{rerank.metrics.hit_rate_at_k:.3f}** (dense + cross-encoder), a "
            f"delta of **+{delta:.3f}**, above the {WIN_MARGIN} margin treated as a real "
            "difference here."
        )
    elif delta > -WIN_MARGIN:
        lines.append(
            f"**A wash on this eval set**: hit-rate@{K} is {dense.metrics.hit_rate_at_k:.3f} "
            f"(dense) vs {rerank.metrics.hit_rate_at_k:.3f} (dense + rerank), within the "
            f"{WIN_MARGIN} margin treated as noise. `apps/api/routers/search.py` still defaults "
            "`rerank=true`: the eval set only measures whether the *ticket itself* ranks in the "
            "top 5, not result *ordering* quality below that, which is where a cross-encoder "
            "pass is expected to matter most in the live search UI."
        )
    else:
        lines.append(
            f"**Dense-only scores higher on this eval set** (delta **{delta:+.3f}**) -- possible "
            "on a small (100-query), synthetic eval set where the correct ticket is almost "
            "always in the dense top-K already, leaving the cross-encoder pass just reshuffling "
            "already-correct results (SPEC M8's hit-rate@5 metric can't see that kind of "
            "reordering)."
        )
    lines.append("")
    return lines


def _render_no_answer_section(rows: list[NoAnswerExample]) -> list[str]:
    lines = [
        '## No-answer behavior (SPEC M8: "RAG endpoint refuses gracefully when retrieval '
        'confidence is low")',
        "",
        f"`POST /tickets/{{id}}/suggested-reply` refuses before ever calling the LLM whenever the "
        f"best retrieved source's cross-encoder score is below `MIN_CONFIDENCE = {MIN_CONFIDENCE}` "
        "(`ml/inference/rag_reply.py`, threshold derivation in `docs/decisions.md`). Demonstrated "
        "here on 3 realistic support queries and 3 clearly off-topic ones, run through the real "
        "retrieval pipeline against the real indexed corpus, not a fixture:",
        "",
        "| Query | Best cross-encoder score | Would refuse? |",
        "|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.query} | {row.best_score:.3f} | {'yes' if row.would_refuse else 'no'} |"
        )
    lines.append("")
    passed = all(row.would_refuse == (row.kind == "off-topic") for row in rows)
    verdict = "PASS" if passed else "FAIL"
    lines.append(
        f"**{verdict}**: every off-topic query refuses and every on-topic query doesn't, on this run."
    )
    lines.append("")
    return lines


def _render_report(
    queries: list[str],
    relevant_ids: list[str],
    dense: VariantResult,
    rerank: VariantResult,
    no_answer_rows: list[NoAnswerExample],
) -> str:
    lines = [
        "# M8 comparison report: dense retrieval vs dense + cross-encoder rerank",
        "",
        "Generated by `scripts/generate_m8_report.py` from `eval_runs` rows persisted during this "
        "run -- every number below comes from a committed eval run (CLAUDE.md rule #5), nothing "
        "here is hand-typed.",
        "",
        f"## Retrieval hit-rate@{K}",
        "",
        "100-query synthetic eval set (`ml/data/retrieval_eval_set.py`): each query is a "
        "resolved ticket's first customer message, the known-relevant answer is that ticket's "
        "own id. See `docs/decisions.md` for why this is a documented limitation (not real "
        "held-out user queries), same shortcut class as this project's other weak-supervision "
        "steps.",
        "",
        "| Variant | hit-rate@5 | Queries |",
        "|---|---|---|",
        f"| dense only (`{dense.model_version}`) | {dense.metrics.hit_rate_at_k:.3f} | "
        f"{dense.metrics.n_queries} |",
        f"| dense + rerank (`{rerank.model_version}`) | {rerank.metrics.hit_rate_at_k:.3f} | "
        f"{rerank.metrics.n_queries} |",
        "",
        *_render_recommendation(dense, rerank),
        "## Example queries",
        "",
        *_example_rows(queries, relevant_ids, dense, rerank),
        "",
        *_render_no_answer_section(no_answer_rows),
    ]
    return "\n".join(lines)


def main() -> None:
    if not EVAL_SET_PATH.exists():
        raise RuntimeError(
            f"{EVAL_SET_PATH} not found -- run `make build-retrieval-eval` first "
            "(docs/m8-how-to-run-locally.md)"
        )
    df = pd.read_parquet(EVAL_SET_PATH)
    queries = df["query"].tolist()
    relevant_ids = df["relevant_ticket_id"].tolist()

    # Lazy: sentence-transformers/chromadb live behind the `search`
    # dependency group, same reason apps/api/routers/search.py defers
    # these imports.
    from ml.inference.embeddings import SentenceEmbeddingPredictor
    from ml.inference.reranker import CrossEncoderReranker

    embedder = SentenceEmbeddingPredictor()
    store = ChromaVectorStore()
    cross_encoder = CrossEncoderReranker()

    dense_ids = run_variant(queries, embedder=embedder, store=store, reranker=None)
    dense = VariantResult(
        "dense", "dense_v1", compute_hit_rate(dense_ids, relevant_ids, k=K), dense_ids
    )
    print(
        f"dense: hit-rate@{K} = {dense.metrics.hit_rate_at_k:.3f} ({dense.metrics.n_queries} queries)"
    )

    rerank_ids = run_variant(queries, embedder=embedder, store=store, reranker=cross_encoder)
    rerank = VariantResult(
        "dense_rerank",
        "dense_rerank_v1",
        compute_hit_rate(rerank_ids, relevant_ids, k=K),
        rerank_ids,
    )
    print(
        f"dense+rerank: hit-rate@{K} = {rerank.metrics.hit_rate_at_k:.3f} "
        f"({rerank.metrics.n_queries} queries)"
    )

    no_answer_rows = run_no_answer_examples(
        NO_ANSWER_EXAMPLES, embedder=embedder, store=store, reranker=cross_encoder
    )
    n_correct = sum(row.would_refuse == (row.kind == "off-topic") for row in no_answer_rows)
    print(
        f"no-answer demonstration: {n_correct}/{len(no_answer_rows)} examples behaved as expected"
    )

    session = SessionLocal()
    try:
        _persist(session, dense)
        _persist(session, rerank)
    finally:
        session.close()

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        _render_report(queries, relevant_ids, dense, rerank, no_answer_rows), encoding="utf-8"
    )
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
