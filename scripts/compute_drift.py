"""Computes and persists SPEC M9's drift signals -- embedding-distribution
distance (centroid cosine shift) and prediction-distribution shift (PSI) --
each run twice: once against real, undisturbed traffic (expected: no
alarm) and once against a simulated, topically-different injected slice
(expected: alarm fires). SPEC M9: "a simulated drift scenario (feed the
app a topically different slice and watch the alarms fire)".

Reference week / live window (docs/decisions.md has the full measurement):
both drawn from the real Twitter corpus's stable high-volume tail
(Oct-Nov 2017, ~3500-3600 tickets/week -- the corpus's early weeks run as
low as 2 tickets/week and would make either signal meaningless).
REFERENCE_WEEK is a single week; LIVE_WINDOW_WEEKS is a multi-week window
(SPEC's own wording distinguishes "a reference week" from "the live
window"), separated by a 2-week gap from the reference week so the two
aren't adjacent.

Embedding vectors: reused directly from M7's offline corpus embeddings
(data/embeddings/tickets_minilm_v1.{npy,parquet}) for every real-corpus
week -- no new embedding computation for the real-traffic scenario. The
simulated scenario's Bitext slice is embedded once here (Bitext was
deliberately excluded from M7's own corpus embeddings, see
docs/decisions.md) -- a one-off batch step, never live/per-request.

Urgency-label distributions (the prediction-drift signal): reused
directly from M5's already-persisted sentiment_trajectory Prediction
payloads (payload["urgency_label"]) for both scenarios -- zero new
inference either way, same reasoning.

Run: uv run python scripts/compute_drift.py
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from api.db.models import AuthorRole, Prediction, Ticket, TicketSource
from api.db.session import SessionLocal
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from ml.evaluation.drift_metrics import (
    EmbeddingDriftResult,
    PredictionDriftResult,
    compute_embedding_drift,
    compute_prediction_drift,
)
from ml.evaluation.metrics import persist_eval_run

ROOT = Path(__file__).resolve().parents[1]
EMBEDDINGS_NPY = ROOT / "data" / "embeddings" / "tickets_minilm_v1.npy"
EMBEDDINGS_PARQUET = ROOT / "data" / "embeddings" / "tickets_minilm_v1.parquet"
REPORT_PATH = ROOT / "docs" / "m9-drift-report.md"

DATASET = "twitter_slice_v1"
EMBEDDING_MODEL_VERSION = "all-MiniLM-L6-v2"
URGENCY_MODEL_VERSION = "transformer_urgency_deberta-v3-small_v1"

# See module docstring + docs/decisions.md: the first week of the dense
# window's stable high-volume tail, and the last 4 weeks of that same
# tail as the "live window" -- a 2-week gap (2017-10-16/10-23/10-30)
# separates the two so they aren't adjacent.
REFERENCE_WEEK = "2017-10-09"
LIVE_WINDOW_WEEKS = ("2017-11-06", "2017-11-13", "2017-11-20", "2017-11-27")

# Sized to roughly match one real week's ticket volume (~3500) so the
# simulated scenario's sample size is comparable to the real one, not
# just directionally different.
BITEXT_EMBEDDING_SAMPLE = 3000


def _week_start(dt: datetime) -> str:
    monday = dt.date() - timedelta(days=dt.weekday())
    return monday.isoformat()


def _load_weekly_twitter_ticket_ids(session: Session) -> dict[str, list[str]]:
    rows = session.execute(
        select(Ticket.id, Ticket.created_at).where(
            Ticket.source == TicketSource.TWITTER, Ticket.created_at.isnot(None)
        )
    ).all()
    by_week: dict[str, list[str]] = defaultdict(list)
    for ticket_id, created_at in rows:
        by_week[_week_start(created_at)].append(str(ticket_id))
    return by_week


def _load_urgency_counts(
    session: Session, ticket_ids: set[str] | None, *, source: TicketSource | None
) -> Counter[str]:
    """ticket_ids=None with a source filter loads every persisted urgency
    label for that source (used for the Bitext simulated scenario, which
    has no created_at to bucket by week); ticket_ids=a set loads only
    those specific tickets' labels (the real reference/live-window
    scenario, source=None since the ids are already Twitter-only)."""
    stmt = (
        select(Ticket.id, Prediction.payload)
        .join(Prediction, Prediction.ticket_id == Ticket.id)
        .where(Prediction.task == "sentiment_trajectory")
    )
    if source is not None:
        stmt = stmt.where(Ticket.source == source)

    counts: Counter[str] = Counter()
    for ticket_id, payload in session.execute(stmt).all():
        if ticket_ids is not None and str(ticket_id) not in ticket_ids:
            continue
        label = payload.get("urgency_label")
        if label:
            counts[label] += 1
    return counts


def _customer_document(ticket: Ticket) -> str | None:
    """Same embedding-unit convention as scripts/compute_embeddings.py:
    the concatenation of a ticket's customer messages only."""
    customer_texts = [m.text_clean for m in ticket.messages if m.author_role == AuthorRole.CUSTOMER]
    if not customer_texts:
        return None
    return "\n".join(customer_texts)


def _embed_bitext_sample(session: Session, limit: int) -> np.ndarray:
    # Lazy: sentence-transformers lives behind the `topics`/`search`
    # dependency groups, not installed by default (same pattern
    # scripts/compute_embeddings.py and scripts/generate_m8_report.py use).
    from ml.inference.embeddings import SentenceEmbeddingPredictor

    stmt = (
        select(Ticket)
        .options(selectinload(Ticket.messages))
        .where(Ticket.source == TicketSource.BITEXT)
        .order_by(Ticket.ingested_at)
        .limit(limit)
    )
    tickets = session.scalars(stmt).all()
    documents = [doc for t in tickets if (doc := _customer_document(t)) is not None]

    predictor = SentenceEmbeddingPredictor()
    return predictor.encode(documents, batch_size=64, show_progress_bar=False)


@dataclass(frozen=True)
class DriftRun:
    embedding_real: EmbeddingDriftResult
    embedding_simulated: EmbeddingDriftResult
    prediction_real: PredictionDriftResult
    prediction_simulated: PredictionDriftResult


def compute_and_persist(session: Session) -> DriftRun:
    embeddings_df = pd.read_parquet(EMBEDDINGS_PARQUET)
    vectors = np.load(EMBEDDINGS_NPY)
    id_to_vector = dict(zip(embeddings_df["ticket_id"], vectors, strict=True))

    by_week = _load_weekly_twitter_ticket_ids(session)
    reference_ids = set(by_week[REFERENCE_WEEK])
    live_real_ids = {tid for week in LIVE_WINDOW_WEEKS for tid in by_week[week]}
    if not reference_ids or not live_real_ids:
        raise RuntimeError(
            f"reference week {REFERENCE_WEEK!r} or live window {LIVE_WINDOW_WEEKS!r} has no "
            "Twitter tickets -- has `make ingest-twitter` (and `make embed-tickets`) run?"
        )

    reference_vectors = np.array([id_to_vector[i] for i in reference_ids if i in id_to_vector])
    live_real_vectors = np.array([id_to_vector[i] for i in live_real_ids if i in id_to_vector])
    print(
        f"embedding drift (real): reference week {REFERENCE_WEEK} n={len(reference_vectors)}, "
        f"live window {LIVE_WINDOW_WEEKS} n={len(live_real_vectors)}"
    )
    embedding_real = compute_embedding_drift(reference_vectors, live_real_vectors)
    persist_eval_run(
        session,
        task="drift_embedding",
        model_version=EMBEDDING_MODEL_VERSION,
        dataset=DATASET,
        split="reference_vs_live_real",
        metrics=embedding_real,
        params={"reference_week": REFERENCE_WEEK, "live_window_weeks": list(LIVE_WINDOW_WEEKS)},
    )
    print(f"  cosine_shift={embedding_real.cosine_shift:.4f} is_alarm={embedding_real.is_alarm}")

    print(f"embedding drift (simulated): embedding {BITEXT_EMBEDDING_SAMPLE} Bitext tickets...")
    bitext_vectors = _embed_bitext_sample(session, BITEXT_EMBEDDING_SAMPLE)
    embedding_simulated = compute_embedding_drift(reference_vectors, bitext_vectors)
    persist_eval_run(
        session,
        task="drift_embedding",
        model_version=EMBEDDING_MODEL_VERSION,
        dataset=DATASET,
        split="reference_vs_live_simulated",
        metrics=embedding_simulated,
        params={
            "reference_week": REFERENCE_WEEK,
            "live_source": "bitext_injection",
            "live_n_requested": BITEXT_EMBEDDING_SAMPLE,
        },
    )
    print(
        f"  cosine_shift={embedding_simulated.cosine_shift:.4f} "
        f"is_alarm={embedding_simulated.is_alarm}"
    )

    reference_urgency = _load_urgency_counts(session, reference_ids, source=None)
    live_real_urgency = _load_urgency_counts(session, live_real_ids, source=None)
    print(
        f"prediction drift (real): reference urgency={dict(reference_urgency)}, "
        f"live urgency={dict(live_real_urgency)}"
    )
    prediction_real = compute_prediction_drift(reference_urgency, live_real_urgency)
    persist_eval_run(
        session,
        task="drift_prediction",
        model_version=URGENCY_MODEL_VERSION,
        dataset=DATASET,
        split="reference_vs_live_real",
        metrics=prediction_real,
        params={"reference_week": REFERENCE_WEEK, "live_window_weeks": list(LIVE_WINDOW_WEEKS)},
    )
    print(f"  psi={prediction_real.psi:.4f} status={prediction_real.status}")

    bitext_urgency = _load_urgency_counts(session, None, source=TicketSource.BITEXT)
    print(f"prediction drift (simulated): bitext urgency={dict(bitext_urgency)}")
    prediction_simulated = compute_prediction_drift(reference_urgency, bitext_urgency)
    persist_eval_run(
        session,
        task="drift_prediction",
        model_version=URGENCY_MODEL_VERSION,
        dataset=DATASET,
        split="reference_vs_live_simulated",
        metrics=prediction_simulated,
        params={"reference_week": REFERENCE_WEEK, "live_source": "bitext_injection"},
    )
    print(f"  psi={prediction_simulated.psi:.4f} status={prediction_simulated.status}")

    return DriftRun(embedding_real, embedding_simulated, prediction_real, prediction_simulated)


def _verdict_row(name: str, real_fires: bool, simulated_fires: bool) -> str:
    passed = (not real_fires) and simulated_fires
    verdict = "PASS" if passed else "FAIL"
    return f"| {name} | {'alarm' if real_fires else 'no alarm'} | {'alarm' if simulated_fires else 'no alarm'} | {verdict} |"


def _render_report(run: DriftRun) -> str:
    lines = [
        "# M9 drift report: embedding-distribution distance + prediction-distribution shift",
        "",
        "Generated by `scripts/compute_drift.py` from `eval_runs` rows persisted during this "
        "run -- every number below comes from a committed eval run (CLAUDE.md rule #5), "
        "nothing here is hand-typed.",
        "",
        f"Reference week: `{REFERENCE_WEEK}`. Live window (real scenario): "
        f"`{', '.join(LIVE_WINDOW_WEEKS)}`. Both drawn from the real Twitter corpus's stable "
        "high-volume tail (~3500-3600 tickets/week) so sample sizes are large enough for a "
        "stable centroid/PSI estimate -- see `docs/decisions.md` for the full measurement that "
        "picked these weeks and the alarm thresholds below.",
        "",
        "## Embedding-distribution distance (centroid cosine shift)",
        "",
        "| Scenario | Reference n | Live n | Cosine shift | Alarm (> "
        f"{run.embedding_real.to_metrics_dict()['threshold']}) |",
        "|---|---|---|---|---|",
        f"| Real (reference week vs. live window) | {run.embedding_real.reference_n} | "
        f"{run.embedding_real.live_n} | {run.embedding_real.cosine_shift:.4f} | "
        f"{run.embedding_real.is_alarm} |",
        f"| Simulated (reference week vs. Bitext-injected slice) | "
        f"{run.embedding_simulated.reference_n} | {run.embedding_simulated.live_n} | "
        f"{run.embedding_simulated.cosine_shift:.4f} | {run.embedding_simulated.is_alarm} |",
        "",
        "## Prediction-distribution shift (urgency-label PSI)",
        "",
        "| Scenario | Reference n | Live n | PSI | Status |",
        "|---|---|---|---|---|",
        f"| Real (reference week vs. live window) | {run.prediction_real.reference_n} | "
        f"{run.prediction_real.live_n} | {run.prediction_real.psi:.4f} | "
        f"{run.prediction_real.status} |",
        f"| Simulated (reference week vs. Bitext-injected slice) | "
        f"{run.prediction_simulated.reference_n} | {run.prediction_simulated.live_n} | "
        f"{run.prediction_simulated.psi:.4f} | {run.prediction_simulated.status} |",
        "",
        '## SPEC M9 acceptance: "a simulated drift scenario... watch the alarms fire"',
        "",
        "| Signal | Real scenario | Simulated scenario | Verdict |",
        "|---|---|---|---|",
        _verdict_row(
            "Embedding distance",
            run.embedding_real.is_alarm,
            run.embedding_simulated.is_alarm,
        ),
        _verdict_row(
            "Prediction shift (PSI)",
            run.prediction_real.status == "alarm",
            run.prediction_simulated.status == "alarm",
        ),
        "",
        "**PASS** means: the real reference-week-vs-live-window comparison does not fire (normal "
        "week-to-week traffic isn't drift), and the simulated topically-different injection does "
        "fire (SPEC M9's demonstration).",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    session = SessionLocal()
    try:
        run = compute_and_persist(session)
    finally:
        session.close()

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(_render_report(run), encoding="utf-8")
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
