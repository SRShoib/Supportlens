import pytest

from ml.inference.base import format_dialogue


def test_joins_turns_as_speaker_colon_text() -> None:
    result = format_dialogue([("Customer", "hi there"), ("Agent", "how can I help?")])
    assert result == "Customer: hi there\nAgent: how can I help?"


def test_single_turn() -> None:
    assert format_dialogue([("Customer", "hello")]) == "Customer: hello"


def test_raises_on_empty_turns() -> None:
    with pytest.raises(ValueError, match="at least one turn"):
        format_dialogue([])
