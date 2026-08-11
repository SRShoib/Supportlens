from ml.data.ner.schema import CharSpan
from ml.data.ner.templates import TEMPLATES, slot_label, template_slots
from ml.data.ner.templates import render as render_template
from ml.inference.rules_ner import ENTITY_LABELS


def test_template_slots_extracts_labels_in_order() -> None:
    assert template_slots("order {ORDER_ID} shipped {DATE}") == ["ORDER_ID", "DATE"]


def test_template_slots_preserves_numbered_suffix() -> None:
    assert template_slots("billed {AMOUNT#1} then {AMOUNT#2}") == ["AMOUNT#1", "AMOUNT#2"]


def test_slot_label_strips_numbered_suffix() -> None:
    assert slot_label("AMOUNT#1") == "AMOUNT"
    assert slot_label("AMOUNT") == "AMOUNT"


def test_render_single_slot() -> None:
    text, spans = render_template("order {ORDER_ID} shipped", {"ORDER_ID": "ORD-99321"})
    assert text == "order ORD-99321 shipped"
    assert spans == [CharSpan(6, 15, "ORDER_ID", "ORD-99321")]


def test_render_multiple_distinct_slots() -> None:
    text, spans = render_template(
        "charged {AMOUNT} for order {ORDER_ID}",
        {"AMOUNT": "$49.99", "ORDER_ID": "ORD-1"},
    )
    assert text == "charged $49.99 for order ORD-1"
    assert spans == [
        CharSpan(8, 14, "AMOUNT", "$49.99"),
        CharSpan(25, 30, "ORDER_ID", "ORD-1"),
    ]


def test_render_repeated_identical_value_gets_correct_independent_offsets() -> None:
    # The bug this design exists to avoid: if offsets were computed by
    # str.find(value) instead of a running cursor, two occurrences of the
    # *same* value would both resolve to the first occurrence's position.
    text, spans = render_template(
        "billed {AMOUNT#1} then {AMOUNT#2}",
        {"AMOUNT#1": "$40", "AMOUNT#2": "$40"},
    )
    assert text == "billed $40 then $40"
    assert spans == [
        CharSpan(7, 10, "AMOUNT", "$40"),
        CharSpan(16, 19, "AMOUNT", "$40"),
    ]
    for span in spans:
        assert text[span.start : span.end] == span.text


def test_render_no_slots() -> None:
    text, spans = render_template("no entities here", {})
    assert text == "no entities here"
    assert spans == []


def test_every_catalog_template_parses_with_known_labels() -> None:
    seen_ids = set()
    for template in TEMPLATES:
        assert template.id not in seen_ids, f"duplicate template id {template.id}"
        seen_ids.add(template.id)

        slots = template_slots(template.text)
        assert slots, f"{template.id} has no slots"
        for slot in slots:
            assert slot_label(slot) in ENTITY_LABELS, f"{template.id}: unknown label in {slot!r}"


def test_every_catalog_template_renders_with_placeholder_values() -> None:
    for template in TEMPLATES:
        slots = template_slots(template.text)
        values = {slot: f"<{slot}_VALUE>" for slot in slots}
        text, spans = render_template(template.text, values)
        assert len(spans) == len(slots)
        for span in spans:
            assert text[span.start : span.end] == span.text
