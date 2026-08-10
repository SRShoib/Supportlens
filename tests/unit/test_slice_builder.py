from datetime import UTC, datetime, timedelta
from pathlib import Path

from ml.data.loaders.twitter import ConversationSummary
from ml.data.slice_builder import SliceConfig, build_slice, load_slice, save_slice

_BASE = datetime(2017, 1, 1, tzinfo=UTC)


def _make_conversation(
    root: str, brand: str, week_offset: int, message_count: int
) -> ConversationSummary:
    return ConversationSummary(
        root_id=root,
        brand=brand,
        start_time=_BASE + timedelta(weeks=week_offset),
        message_count=message_count,
        tweet_ids=frozenset({root}),
    )


def _big_and_small_brand_corpus() -> dict[str, ConversationSummary]:
    conversations = {}
    for i in range(200):
        root = f"big-{i}"
        conversations[root] = _make_conversation(root, "BigBrand", i % 20, message_count=5)
    for i in range(20):
        root = f"small-{i}"
        conversations[root] = _make_conversation(root, "SmallBrand", i % 20, message_count=3)
    return conversations


def test_brand_cap_is_respected_when_all_brands_have_enough_data() -> None:
    conversations = {}
    for i in range(200):
        root = f"big-{i}"
        conversations[root] = _make_conversation(root, "BigBrand", i % 20, message_count=5)
    for i in range(200):
        root = f"other-{i}"
        conversations[root] = _make_conversation(root, "OtherBrand", i % 20, message_count=5)

    result = build_slice(
        conversations, SliceConfig(target_messages=1000, max_brand_share=0.5, seed=42)
    )

    total = sum(result.brand_message_counts.values())
    assert result.brand_message_counts["BigBrand"] / total <= 0.5 + 1e-9


def test_brand_cap_applies_to_target_allocation_not_realized_share() -> None:
    """A capped brand's realized share can exceed the cap if another brand's
    conversation pool runs dry before it reaches its own allotted target."""
    conversations = _big_and_small_brand_corpus()
    result = build_slice(
        conversations, SliceConfig(target_messages=1000, max_brand_share=0.5, seed=42)
    )

    small_brand_pool = sum(
        c.message_count for c in conversations.values() if c.brand == "SmallBrand"
    )
    assert result.brand_message_counts["SmallBrand"] == small_brand_pool
    assert result.brand_message_counts["BigBrand"] == 500


def test_message_target_within_tolerance_when_data_available() -> None:
    conversations = {
        f"c-{i}": _make_conversation(f"c-{i}", "OnlyBrand", i % 10, message_count=10)
        for i in range(500)
    }
    result = build_slice(
        conversations, SliceConfig(target_messages=1000, max_brand_share=1.0, seed=42)
    )

    total = sum(result.brand_message_counts.values())
    assert abs(total - 1000) <= 10


def test_selection_spreads_across_weekly_buckets() -> None:
    conversations = {
        f"c-{i}": _make_conversation(f"c-{i}", "OnlyBrand", i % 20, message_count=5)
        for i in range(400)
    }
    result = build_slice(
        conversations, SliceConfig(target_messages=1000, max_brand_share=1.0, seed=42)
    )

    weeks_represented = {
        conversations[r].start_time.isocalendar()[1] for r in result.selected_roots
    }
    assert len(weeks_represented) >= 15


def test_seed_42_reproducible_across_runs() -> None:
    conversations = _big_and_small_brand_corpus()
    config = SliceConfig(target_messages=500, max_brand_share=0.5, seed=42)

    result1 = build_slice(conversations, config)
    result2 = build_slice(conversations, config)

    assert result1.selected_roots == result2.selected_roots


def test_different_seed_can_change_selection() -> None:
    conversations = _big_and_small_brand_corpus()
    result_a = build_slice(
        conversations, SliceConfig(target_messages=200, max_brand_share=0.5, seed=42)
    )
    result_b = build_slice(
        conversations, SliceConfig(target_messages=200, max_brand_share=0.5, seed=7)
    )

    assert result_a.selected_roots != result_b.selected_roots


def test_empty_corpus_returns_empty_slice() -> None:
    result = build_slice({}, SliceConfig())
    assert result.selected_roots == frozenset()


def test_manifest_roundtrips_through_parquet(tmp_path: Path) -> None:
    conversations = {
        "a": _make_conversation("a", "X", 0, message_count=3),
        "b": _make_conversation("b", "X", 1, message_count=3),
    }
    result = build_slice(
        conversations, SliceConfig(target_messages=10, max_brand_share=1.0, seed=42)
    )

    path = tmp_path / "slice.parquet"
    save_slice(result, path)
    loaded = load_slice(path)

    assert loaded == result.selected_roots
    assert (tmp_path / "slice.parquet.json").exists()
