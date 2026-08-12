import pytest

from ml.evaluation.trend_metrics import (
    MIN_SPIKE_TICKETS,
    Z_SCORE_THRESHOLD,
    WeeklyTopicVolume,
    compute_topic_trends,
    detect_emerging_topics,
    select_dense_window,
)

# 6 consecutive Mondays -- long enough that every week still has
# MIN_HISTORY_WEEKS=4 "other" weeks to compare against.
WEEKS = [
    "2024-01-01",
    "2024-01-08",
    "2024-01-15",
    "2024-01-22",
    "2024-01-29",
    "2024-02-05",
]
SPIKE_WEEK = WEEKS[-1]


def _rows_for(rows: list[WeeklyTopicVolume], topic_id: int) -> dict[str, WeeklyTopicVolume]:
    return {r.week: r for r in rows if r.topic_id == topic_id}


def test_injected_spike_is_flagged_as_emerging() -> None:
    # A large stable background topic supplies most of the weekly volume,
    # so topic 100's own spike moves its *share*, not just its raw count.
    weekly_counts = {
        100: dict.fromkeys(WEEKS[:-1], 5) | {SPIKE_WEEK: 40},
        200: dict.fromkeys(WEEKS, 100),
    }

    rows = compute_topic_trends(weekly_counts, WEEKS)

    spike_row = _rows_for(rows, 100)[SPIKE_WEEK]
    assert spike_row.z_score is not None
    assert spike_row.z_score > Z_SCORE_THRESHOLD
    assert spike_row.is_emerging is True


def test_flat_share_never_fires() -> None:
    weekly_counts = {
        300: dict.fromkeys(WEEKS, 30),
        400: dict.fromkeys(WEEKS, 100),
    }

    rows = compute_topic_trends(weekly_counts, WEEKS)

    for row in _rows_for(rows, 300).values():
        assert row.z_score == pytest.approx(0.0)
        assert row.is_emerging is False


def test_short_history_is_skipped_not_flagged() -> None:
    # Only 4 weeks total -> every week has just 3 "other" weeks, below
    # MIN_HISTORY_WEEKS=4, regardless of how dramatic the counts are.
    short_weeks = WEEKS[:4]
    weekly_counts = {
        500: dict.fromkeys(short_weeks[:-1], 5) | {short_weeks[-1]: 500},
    }

    rows = compute_topic_trends(weekly_counts, short_weeks)

    for row in rows:
        assert row.z_score is None
        assert row.is_emerging is False


def test_min_spike_tickets_blocks_the_zero_history_single_ticket_case() -> None:
    # topic 600 is silent every week except one lone ticket in the last
    # week -- leave-one-out stdev over an all-zero history is 0, which
    # would otherwise blow the z-score up arbitrarily large.
    weekly_counts = {
        600: dict.fromkeys(WEEKS[:-1], 0) | {SPIKE_WEEK: 1},
        700: dict.fromkeys(WEEKS, 50),
    }

    rows = compute_topic_trends(weekly_counts, WEEKS)

    spike_row = _rows_for(rows, 600)[SPIKE_WEEK]
    assert spike_row.count < MIN_SPIKE_TICKETS
    assert spike_row.is_emerging is False


def test_global_volume_rise_does_not_fire_when_share_is_unchanged() -> None:
    # topic 800's raw count doubles in the last week, but so does every
    # other topic's -- its *share* of total volume never moves, so this
    # must not be flagged as an emerging issue.
    weekly_counts = {
        800: dict.fromkeys(WEEKS[:-1], 10) | {SPIKE_WEEK: 20},
        900: dict.fromkeys(WEEKS[:-1], 90) | {SPIKE_WEEK: 180},
    }

    rows = compute_topic_trends(weekly_counts, WEEKS)

    spike_row = _rows_for(rows, 800)[SPIKE_WEEK]
    assert spike_row.count == 20
    assert spike_row.z_score == pytest.approx(0.0)
    assert spike_row.is_emerging is False


def test_detect_emerging_topics_filters_to_flagged_rows_only() -> None:
    weekly_counts = {
        100: dict.fromkeys(WEEKS[:-1], 5) | {SPIKE_WEEK: 40},
        200: dict.fromkeys(WEEKS, 100),
    }

    rows = compute_topic_trends(weekly_counts, WEEKS)
    emerging = detect_emerging_topics(rows)

    assert emerging
    assert all(row.is_emerging for row in emerging)
    assert (100, SPIKE_WEEK) in {(row.topic_id, row.week) for row in emerging}


def test_select_dense_window_drops_sparse_weeks_below_the_density_threshold() -> None:
    total_by_week = {
        "2017-01-02": 3,  # sparse historical noise, far below the median
        "2023-01-02": 200,
        "2023-01-09": 210,
        "2023-01-16": 190,
        "2023-01-23": 205,
        "2023-01-30": 195,
    }

    window = select_dense_window(total_by_week)

    assert window == ["2023-01-02", "2023-01-09", "2023-01-16", "2023-01-23", "2023-01-30"]


def test_select_dense_window_returns_longest_run_when_two_dense_blocks_exist() -> None:
    total_by_week = {
        "2023-01-02": 200,
        "2023-01-09": 210,
        "2023-01-16": 190,
        # gap
        "2023-02-06": 205,
    }

    window = select_dense_window(total_by_week)

    assert window == ["2023-01-02", "2023-01-09", "2023-01-16"]


def test_select_dense_window_empty_input_returns_empty_list() -> None:
    assert select_dense_window({}) == []


def test_select_dense_window_all_zero_returns_empty_list() -> None:
    assert select_dense_window({"2023-01-02": 0, "2023-01-09": 0}) == []
