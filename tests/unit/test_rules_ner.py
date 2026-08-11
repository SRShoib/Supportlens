import pytest

from ml.inference.base import EntitySpan
from ml.inference.rules_ner import RulesEntityPredictor, extract_spans, resolve_overlaps, trim_span


def _labels(text: str) -> list[tuple[str, str]]:
    return [(s.label, s.text) for s in extract_spans(text)]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("order #99321 has arrived", [("ORDER_ID", "99321")]),
        ("your order number 12345 has shipped", [("ORDER_ID", "12345")]),
        ("confirmation: ORD-99321 received", [("ORDER_ID", "ORD-99321")]),
        ("tracking 1Z999AA10123456784 never updated", [("ORDER_ID", "1Z999AA10123456784")]),
        ("my order is with the courier", []),  # no id present at all
    ],
)
def test_order_id_rules(text: str, expected: list[tuple[str, str]]) -> None:
    assert _labels(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("account number 4455-9911 was debited", [("ACCOUNT_REF", "4455-9911")]),
        ("my case ref CASE99 is still open", [("ACCOUNT_REF", "CASE99")]),
        ("card ending in 4432", [("ACCOUNT_REF", "4432")]),
        ("last 4 digits are 1234", [("ACCOUNT_REF", "1234")]),
        # The false-positive this rule is specifically guarded against: no
        # digit anywhere in the candidate token means it isn't an id.
        ("account number is closed", []),
        ("fix my account please", []),
    ],
)
def test_account_ref_rules(text: str, expected: list[tuple[str, str]]) -> None:
    assert _labels(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("charged $49.99 for the order", [("AMOUNT", "$49.99")]),
        ("that cost £12.50 in total", [("AMOUNT", "£12.50")]),
        ("paid USD 40 for shipping", [("AMOUNT", "USD 40")]),
        ("it was 40 dollars total", [("AMOUNT", "40 dollars")]),
        ("they refunded me 49.99 in total", [("AMOUNT", "49.99")]),
        # A bare number with no monetary governor is not an amount.
        ("I ordered 3 items", []),
        ("5000 miles were credited", []),
    ],
)
def test_amount_rules(text: str, expected: list[tuple[str, str]]) -> None:
    assert _labels(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("it broke yesterday", [("DATE", "yesterday")]),
        ("shipped on March 3", [("DATE", "March 3")]),
        ("delivered 03/12/2023", [("DATE", "03/12/2023")]),
        ("ordered two weeks ago", [("DATE", "two weeks ago")]),
        ("still broken 3 days back", [("DATE", "3 days back")]),
        ("nothing since Black Friday", [("DATE", "Black Friday")]),
        # A pure duration that doesn't fix a point in time is not a DATE.
        ("I waited 3 days for a reply", []),
        ("a 2 hour wait is unacceptable", []),
    ],
)
def test_date_rules(text: str, expected: list[tuple[str, str]]) -> None:
    assert _labels(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("my iPhone 12 Pro Max screen cracked", [("PRODUCT", "iPhone 12 Pro Max")]),
        ("upgrade to Xbox Game Pass please", [("PRODUCT", "Xbox Game Pass")]),
        ("lowercase iphone 12 pro still matches", [("PRODUCT", "iphone 12 pro")]),
        # Brand alone, addressing the vendor, is not a PRODUCT mention.
        ("<USER> this is unacceptable", []),
    ],
)
def test_product_rules(text: str, expected: list[tuple[str, str]]) -> None:
    assert _labels(text) == expected


def test_every_span_offset_matches_its_text() -> None:
    text = "order ORD-99321 for iPhone 12 Pro Max on March 3 charged $1,299.00, ref ACC-482910"
    for span in extract_spans(text):
        assert text[span.start : span.end] == span.text


def test_multiple_entities_in_one_sentence_all_found() -> None:
    text = "order ORD-12345 shipped March 3, charged $9.99, ref ACC-1234"
    labels = {label for label, _ in _labels(text)}
    assert labels == {"ORDER_ID", "DATE", "AMOUNT", "ACCOUNT_REF"}


def test_no_span_has_leading_or_trailing_whitespace_or_punctuation() -> None:
    for span in extract_spans("charged $49.99, ref ACC-1234."):
        assert span.text == span.text.strip()
        assert span.text[-1] not in ",."


def test_known_weakness_rule_based_list_enumeration() -> None:
    # Documented honest limitation (docs/m4-rules-vs-model-report.md): only
    # the first id after the trigger word is caught, since the pattern
    # requires the trigger word itself, not just proximity to one.
    assert _labels("orders 1111, 2222 and 3333 never arrived") == [("ORDER_ID", "1111")]


class TestTrimSpan:
    def test_strips_surrounding_whitespace(self) -> None:
        text = "  99321  "
        assert trim_span(text, 0, len(text)) == (2, 7)

    def test_strips_trailing_punctuation(self) -> None:
        text = "order #99321."
        assert trim_span(text, 6, 13) == (6, 12)

    def test_all_punctuation_collapses_to_empty_span(self) -> None:
        text = "..."
        assert trim_span(text, 0, 3) == (0, 0)


class TestResolveOverlaps:
    def test_longest_span_wins_over_shorter_overlapping_span(self) -> None:
        short = EntitySpan(start=0, end=5, label="ORDER_ID", text="ORD-1", score=1.0)
        long = EntitySpan(start=0, end=9, label="ORDER_ID", text="ORD-12345", score=1.0)
        assert resolve_overlaps([short, long]) == [long]

    def test_equal_length_overlap_ties_break_by_input_order(self) -> None:
        first = EntitySpan(start=0, end=5, label="ORDER_ID", text="ORD-1", score=1.0)
        second = EntitySpan(start=0, end=5, label="ACCOUNT_REF", text="ORD-1", score=1.0)
        assert resolve_overlaps([first, second]) == [first]
        assert resolve_overlaps([second, first]) == [second]

    def test_non_overlapping_spans_both_kept_sorted_by_position(self) -> None:
        later = EntitySpan(start=10, end=14, label="DATE", text="today", score=1.0)
        earlier = EntitySpan(start=0, end=4, label="AMOUNT", text="$40", score=1.0)
        assert resolve_overlaps([later, earlier]) == [earlier, later]

    def test_no_spans_returns_empty(self) -> None:
        assert resolve_overlaps([]) == []


class TestRulesEntityPredictor:
    def test_predict_returns_one_result_per_text(self) -> None:
        predictor = RulesEntityPredictor()
        results = predictor.predict(["charged $40 today", "no entities here"])
        assert len(results) == 2

    def test_predict_empty_list_returns_empty(self) -> None:
        assert RulesEntityPredictor().predict([]) == []

    def test_predict_entities_match_extract_spans(self) -> None:
        text = "order ORD-99321 charged $49.99"
        result = RulesEntityPredictor().predict([text])[0]
        assert result.entities == extract_spans(text)
        assert result.truncated is False
