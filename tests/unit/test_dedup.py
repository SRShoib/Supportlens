from dataclasses import dataclass

from ml.data.dedup import content_hash, dedup_messages


@dataclass
class _Msg:
    id: str
    text_clean: str


def test_golden_hash_value() -> None:
    assert content_hash("hello world") == "2aae6c35c94fcfb415dbe95f408b9ce91ee846ed"


def test_case_and_whitespace_variant_same_hash() -> None:
    assert content_hash("hello world") == content_hash("  Hello   World  ")


def test_mask_token_variant_collides() -> None:
    """Two messages differing only in which entity got masked hash the same."""
    assert content_hash("<USER> thanks so much") == content_hash("<EMAIL> thanks so much")


def test_different_text_different_hash() -> None:
    assert content_hash("hello world") != content_hash("goodbye world")


def test_exact_duplicate_removed_first_wins() -> None:
    msgs = [_Msg("1", "hello there"), _Msg("2", "hello there"), _Msg("3", "unique")]
    result = list(dedup_messages(msgs))
    assert [m.id for m in result] == ["1", "3"]


def test_case_whitespace_variant_removed() -> None:
    msgs = [_Msg("1", "Hello   There"), _Msg("2", "hello there")]
    result = list(dedup_messages(msgs))
    assert [m.id for m in result] == ["1"]


def test_no_duplicates_keeps_all() -> None:
    msgs = [_Msg("1", "a"), _Msg("2", "b"), _Msg("3", "c")]
    result = list(dedup_messages(msgs))
    assert [m.id for m in result] == ["1", "2", "3"]


def test_empty_iterable() -> None:
    assert list(dedup_messages([])) == []
