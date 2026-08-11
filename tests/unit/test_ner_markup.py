import pytest

from ml.data.ner.markup import MarkupError, parse, render
from ml.data.ner.schema import CharSpan


def test_render_no_entities() -> None:
    assert render("hello world", []) == "hello world"


def test_render_single_entity() -> None:
    text = "order ORD-99321 shipped"
    spans = [CharSpan(6, 15, "ORDER_ID", "ORD-99321")]
    assert render(text, spans) == "order [ORD-99321|ORDER_ID] shipped"


def test_render_multiple_entities_unsorted_input() -> None:
    text = "charged $49.99 for iPhone 12 Pro Max"
    spans = [
        CharSpan(19, 37, "PRODUCT", "iPhone 12 Pro Max"),
        CharSpan(8, 14, "AMOUNT", "$49.99"),
    ]
    assert render(text, spans) == "charged [$49.99|AMOUNT] for [iPhone 12 Pro Max|PRODUCT]"


def test_render_escapes_brackets_and_pipes_and_backslashes_in_plain_text() -> None:
    text = r"check [order] status | see \notes"
    assert render(text, []) == r"check \[order\] status \| see \\notes"


def test_render_escapes_embedded_newlines() -> None:
    # Real message text can contain literal newlines (confirmed on the
    # gold-set candidate pool) -- an unescaped one would silently split a
    # single markdown entry across multiple lines and corrupt
    # ml/data/ner/gold.py's line-based parse_gold_markdown().
    text = "line one\nline two"
    markup = render(text, [])
    assert "\n" not in markup
    assert markup == "line one\\nline two"


def test_render_rejects_overlapping_spans() -> None:
    text = "order ORD-99321 shipped"
    spans = [CharSpan(6, 15, "ORDER_ID", "ORD-99321"), CharSpan(10, 20, "ACCOUNT_REF", "99321 sh")]
    with pytest.raises(MarkupError, match="overlapping"):
        render(text, spans)


def test_parse_no_entities() -> None:
    assert parse("hello world") == ("hello world", [])


def test_parse_single_entity() -> None:
    text, spans = parse("order [ORD-99321|ORDER_ID] shipped")
    assert text == "order ORD-99321 shipped"
    assert spans == [CharSpan(6, 15, "ORDER_ID", "ORD-99321")]


def test_parse_multiple_entities() -> None:
    text, spans = parse("charged [$49.99|AMOUNT] for [iPhone 12 Pro Max|PRODUCT]")
    assert text == "charged $49.99 for iPhone 12 Pro Max"
    assert spans == [
        CharSpan(8, 14, "AMOUNT", "$49.99"),
        CharSpan(19, 36, "PRODUCT", "iPhone 12 Pro Max"),
    ]


def test_parse_unescapes_brackets_and_pipes_and_backslashes() -> None:
    text, spans = parse(r"check \[order\] status \| see \\notes")
    assert text == r"check [order] status | see \notes"
    assert spans == []


def test_parse_unescapes_embedded_newlines() -> None:
    text, spans = parse("line one\\nline two")
    assert text == "line one\nline two"
    assert spans == []


def test_parse_rejects_unknown_label() -> None:
    with pytest.raises(MarkupError, match="unknown label"):
        parse("order [ORD-99321|NOT_A_LABEL] shipped")


def test_parse_rejects_missing_pipe() -> None:
    with pytest.raises(MarkupError, match="missing '\\|'"):
        parse("order [ORD-99321 ORDER_ID] shipped")


def test_parse_rejects_unterminated_block() -> None:
    with pytest.raises(MarkupError, match="unterminated"):
        parse("order [ORD-99321|ORDER_ID shipped")


def test_parse_rejects_unmatched_closing_bracket() -> None:
    with pytest.raises(MarkupError, match="unmatched"):
        parse("order ORD-99321] shipped")


def test_parse_rejects_empty_surface() -> None:
    with pytest.raises(MarkupError, match="empty entity surface"):
        parse("order [|ORDER_ID] shipped")


def test_parse_rejects_nested_brackets() -> None:
    with pytest.raises(MarkupError):
        parse("order [[ORD-99321]|ORDER_ID] shipped")


@pytest.mark.parametrize(
    ("text", "spans"),
    [
        ("no entities at all", []),
        ("order ORD-99321 shipped", [CharSpan(6, 15, "ORDER_ID", "ORD-99321")]),
        (
            "charged $49.99 for iPhone 12 Pro Max on March 3, ref ACC-1234",
            [
                CharSpan(8, 14, "AMOUNT", "$49.99"),
                CharSpan(19, 36, "PRODUCT", "iPhone 12 Pro Max"),
                CharSpan(40, 47, "DATE", "March 3"),
                CharSpan(53, 61, "ACCOUNT_REF", "ACC-1234"),
            ],
        ),
        (r"literal [brackets] and | pipes and \backslashes", []),
        (
            "outage..we had one a few days ago.. strange\n\n<URL>",
            [CharSpan(15, 33, "DATE", "one a few days ago")],
        ),
    ],
)
def test_render_parse_round_trip(text: str, spans: list[CharSpan]) -> None:
    markup = render(text, spans)
    parsed_text, parsed_spans = parse(markup)
    assert parsed_text == text
    assert parsed_spans == spans


def test_parse_render_round_trip_on_canonical_markup() -> None:
    markup = r"charged [$49.99|AMOUNT] for stuff \[in brackets\]"
    text, spans = parse(markup)
    assert render(text, spans) == markup
