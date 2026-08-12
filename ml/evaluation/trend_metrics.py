"""Weekly topic-volume trend detection (SPEC M7: "topic volume per week,
flag topics whose volume z-score > 2"). Pure stdlib -- no DB, no pandas,
no numpy -- so this is importable and fully testable without the `topics`
dependency group installed (ml/inference/embeddings.py and
ml/training/topic_model.py are never on the API's or CI's default import
path; only this module and topic_metrics.py are).

`week` throughout is an ISO date string ("YYYY-MM-DD") naming the Monday a
week bucket starts on -- callers are responsible for producing Monday-
aligned buckets (e.g. Postgres `date_trunc('week', tickets.created_at)`)
before calling into this module.

Two traps a naive z-score falls into on this project's real data, both
handled here (see docs/decisions.md for the full writeup):

1. **Self-inflation.** Computing a week's z-score from a mean/stdev that
   *includes* that week lets one huge spike inflate its own stdev and
   suppress the z-score it's supposed to trigger. `_leave_one_out_zscore`
   uses only the topic's *other* weeks.
2. **Global-volume confound.** If total ticket volume across every topic
   doubles in one week, every topic's raw count rises together -- that's
   the whole queue getting busier, not an "emerging issue". Detection runs
   on each topic's *share* of that week's total volume, not its raw count.

`MIN_HISTORY_WEEKS` and `MIN_SPIKE_TICKETS` are both required gates, not
tie-breakers -- see their docstrings.

`select_dense_window` handles a separate, corpus-specific trap: the twcs
corpus spans 2008-05 to 2017-12 but is 99.6% concentrated in ~10 weeks of
Sep-Dec 2017 (docs/decisions.md). Without discarding the ~480 near-empty
week buckets outside that window first, every z-score below is computed
over mostly-zero history and is meaningless.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import date, timedelta
from itertools import pairwise

MIN_HISTORY_WEEKS = 4
"""A topic needs at least this many *other* weeks of history before a
z-score means anything; below this, compute_topic_trends reports
z_score=None and is_emerging=False for that (topic, week) rather than a
z-score computed from 1-3 data points."""

MIN_SPIKE_TICKETS = 5
"""Guards the degenerate case where a topic's weekly counts are mostly
zero (e.g. 0, 0, 0, 1): the leave-one-out stdev over an all-zero history is
0, so even one ticket produces an enormous z-score. Requiring raw ticket
volume (not just share) also stops a topic firing purely because overall
traffic collapsed and its tiny constant count became a larger share."""

Z_SCORE_THRESHOLD = 2.0
"""SPEC M7's literal threshold: "flag topics whose volume z-score > 2"."""

DENSE_WEEK_MIN_SHARE_OF_MEDIAN = 0.10
"""A week bucket must carry at least this fraction of the median non-empty
week's total ticket volume to count as part of the dense analysis window
-- see select_dense_window."""

_ZERO_VARIANCE_EPSILON = 1e-9
"""Denominator floor for a leave-one-out stdev of exactly 0 (a perfectly
flat history). Without this, any deviation from a flat history divides by
zero; with it, the z-score is a large-but-finite, JSON-serializable number,
and MIN_SPIKE_TICKETS is what actually decides whether it's real."""


@dataclass(frozen=True)
class WeeklyTopicVolume:
    topic_id: int
    week: str
    count: int
    share: float
    z_score: float | None
    is_emerging: bool


def _next_week(week: str) -> str:
    return (date.fromisoformat(week) + timedelta(days=7)).isoformat()


def select_dense_window(total_by_week: dict[str, int]) -> list[str]:
    """Returns the longest contiguous run of weeks (ascending) whose total
    volume is at least DENSE_WEEK_MIN_SHARE_OF_MEDIAN of the median
    non-empty week's volume. Empty input returns []. "Contiguous" means
    each week is exactly 7 days after the previous one -- a gap breaks the
    run even if both weeks individually clear the density threshold."""
    non_empty = [count for count in total_by_week.values() if count > 0]
    if not non_empty:
        return []

    threshold = statistics.median(non_empty) * DENSE_WEEK_MIN_SHARE_OF_MEDIAN
    dense_weeks = sorted(week for week, count in total_by_week.items() if count >= threshold)
    if not dense_weeks:
        return []

    best_run = [dense_weeks[0]]
    current_run = [dense_weeks[0]]
    for prev_week, week in pairwise(dense_weeks):
        if _next_week(prev_week) != week:
            current_run = []
        current_run.append(week)
        if len(current_run) > len(best_run):
            best_run = current_run
    return best_run


def _leave_one_out_zscore(value: float, others: list[float]) -> float:
    mean = statistics.mean(others)
    stdev = statistics.stdev(others) or _ZERO_VARIANCE_EPSILON
    return (value - mean) / stdev


def compute_topic_trends(
    weekly_counts: dict[int, dict[str, int]], weeks: list[str]
) -> list[WeeklyTopicVolume]:
    """weekly_counts[topic_id][week] -> ticket count for that topic in that
    week; a topic/week combination absent from the inner dict is treated as
    0. `weeks` is the caller's chosen analysis window (typically
    select_dense_window's output) -- every topic is evaluated against
    exactly this set of weeks, in this order, regardless of what weeks
    happen to be present in weekly_counts.

    Returns one WeeklyTopicVolume per (topic, week) pair -- len(weeks) rows
    per topic in weekly_counts, in topic-then-week order.
    """
    total_by_week = {
        week: sum(counts.get(week, 0) for counts in weekly_counts.values()) for week in weeks
    }

    rows: list[WeeklyTopicVolume] = []
    for topic_id in sorted(weekly_counts):
        counts = weekly_counts[topic_id]
        shares = {
            week: (counts.get(week, 0) / total_by_week[week] if total_by_week[week] else 0.0)
            for week in weeks
        }
        for week in weeks:
            count = counts.get(week, 0)
            share = shares[week]
            other_shares = [shares[w] for w in weeks if w != week]

            z_score: float | None = None
            is_emerging = False
            if len(other_shares) >= MIN_HISTORY_WEEKS:
                z_score = _leave_one_out_zscore(share, other_shares)
                is_emerging = z_score > Z_SCORE_THRESHOLD and count >= MIN_SPIKE_TICKETS

            rows.append(WeeklyTopicVolume(topic_id, week, count, share, z_score, is_emerging))
    return rows


def detect_emerging_topics(rows: list[WeeklyTopicVolume]) -> list[WeeklyTopicVolume]:
    return [row for row in rows if row.is_emerging]
