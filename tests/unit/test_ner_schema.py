from pathlib import Path

import pytest

from ml.data.ner.schema import (
    CharSpan,
    NerExample,
    NerValidationError,
    append_jsonl,
    read_jsonl,
    validate_example,
    write_jsonl,
)


def _example(text: str, entities: list[CharSpan], **overrides: object) -> NerExample:
    defaults: dict[str, object] = {
        "id": "tpl:000001",
        "text": text,
        "entities": entities,
        "source": "template",
        "split": "train",
    }
    defaults.update(overrides)
    return NerExample(**defaults)  # type: ignore[arg-type]


def test_valid_example_passes() -> None:
    text = "order ORD-99321 shipped yesterday"
    example = _example(text, [CharSpan(6, 15, "ORDER_ID", "ORD-99321")])
    validate_example(example)  # does not raise


def test_valid_example_with_no_entities_passes() -> None:
    validate_example(_example("no entities in this message", []))


def test_rejects_span_text_not_matching_offsets() -> None:
    text = "order ORD-99321 shipped yesterday"
    example = _example(text, [CharSpan(6, 15, "ORDER_ID", "WRONG-VALUE")])
    with pytest.raises(NerValidationError, match="does not match"):
        validate_example(example)


def test_rejects_non_fixed_point_text() -> None:
    example = _example("check &lt;this&gt; out", [])
    with pytest.raises(NerValidationError, match="fixed point"):
        validate_example(example)


def test_rejects_overlapping_spans() -> None:
    text = "order ORD-99321 shipped yesterday"
    example = _example(
        text,
        [
            CharSpan(6, 15, "ORDER_ID", "ORD-99321"),
            CharSpan(10, 20, "ACCOUNT_REF", "99321 ship"),
        ],
    )
    with pytest.raises(NerValidationError, match="overlapping"):
        validate_example(example)


def test_rejects_zero_length_span() -> None:
    text = "order ORD-99321 shipped yesterday"
    example = _example(text, [CharSpan(6, 6, "ORDER_ID", "")])
    with pytest.raises(NerValidationError, match="zero/negative-length"):
        validate_example(example)


def test_rejects_whitespace_padded_span() -> None:
    # Text itself stays single-spaced (a clean_text fixed point) so this
    # exercises the whitespace-padded-span check specifically, not the
    # fixed-point check above it.
    text = "order ORD-99321 shipped yesterday"
    example = _example(text, [CharSpan(5, 15, "ORDER_ID", " ORD-99321")])
    with pytest.raises(NerValidationError, match="whitespace-padded"):
        validate_example(example)


def test_rejects_unknown_label() -> None:
    text = "order ORD-99321 shipped yesterday"
    example = _example(text, [CharSpan(6, 15, "NOT_A_REAL_LABEL", "ORD-99321")])
    with pytest.raises(NerValidationError, match="unknown label"):
        validate_example(example)


def test_rejects_out_of_bounds_span() -> None:
    text = "short text"
    example = _example(text, [CharSpan(0, 999, "ORDER_ID", "short text")])
    with pytest.raises(NerValidationError, match="out of bounds"):
        validate_example(example)


def test_jsonl_round_trip(tmp_path: Path) -> None:
    examples = [
        _example(
            "order ORD-99321 shipped yesterday",
            [CharSpan(6, 15, "ORDER_ID", "ORD-99321")],
            id="tpl:000001",
        ),
        _example(
            "charged $49.99 for iPhone 12 Pro Max",
            [
                CharSpan(8, 14, "AMOUNT", "$49.99"),
                CharSpan(19, 37, "PRODUCT", "iPhone 12 Pro Max"),
            ],
            id="tpl:000002",
            source="template",
            split="val",
            template_id="amount_product_v1",
        ),
    ]
    path = tmp_path / "ner_v1.jsonl"
    write_jsonl(examples, path)
    assert read_jsonl(path) == examples


def test_write_jsonl_creates_parent_directories(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dir" / "ner_v1.jsonl"
    write_jsonl([_example("hello world", [])], path)
    assert path.exists()


def test_append_jsonl_adds_to_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "ner_v1_paraphrase.jsonl"
    first = _example("hello world", [], id="para:000001")
    second = _example("goodbye world", [], id="para:000002")
    append_jsonl([first], path)
    append_jsonl([second], path)
    assert read_jsonl(path) == [first, second]


def test_read_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "ner_v1.jsonl"
    write_jsonl([_example("hello world", [])], path)
    with path.open("a", encoding="utf-8") as f:
        f.write("\n\n")
    assert len(read_jsonl(path)) == 1
