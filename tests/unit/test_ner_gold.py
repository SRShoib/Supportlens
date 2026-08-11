import random
from dataclasses import dataclass
from pathlib import Path

import pytest

from ml.data.ner.gold import (
    GoldCandidate,
    GoldImportError,
    GoldSetError,
    GoldSetReport,
    import_gold_markdown,
    parse_gold_markdown,
    propose_entities,
    render_gold_markdown,
    select_gold_candidates,
    validate_gold_set,
)
from ml.data.ner.schema import CharSpan, NerExample, NerValidationError, read_jsonl

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "ner_gold_sample.jsonl"


def _load_fixture() -> list[NerExample]:
    return read_jsonl(FIXTURE_PATH)


def test_fixture_loads_five_examples() -> None:
    assert len(_load_fixture()) == 5


def test_valid_gold_set_passes_with_matching_expected_size() -> None:
    report = validate_gold_set(_load_fixture(), expected_size=5, min_spans_per_type=1)
    assert isinstance(report, GoldSetReport)
    assert report.n_examples == 5


def test_wrong_size_raises() -> None:
    with pytest.raises(GoldSetError, match="expected exactly"):
        validate_gold_set(_load_fixture(), expected_size=200)


def test_duplicate_ids_raise() -> None:
    examples = _load_fixture()
    duplicated = [examples[0], examples[0], *examples[1:]]
    with pytest.raises(GoldSetError, match="duplicate example ids"):
        validate_gold_set(duplicated, expected_size=len(duplicated), min_spans_per_type=1)


def test_overlap_with_synthetic_ids_raises() -> None:
    examples = _load_fixture()
    with pytest.raises(GoldSetError, match="overlap the synthetic training set"):
        validate_gold_set(
            examples,
            expected_size=5,
            min_spans_per_type=1,
            synthetic_ids=["gold:0001", "tpl:000042"],
        )


def test_no_overlap_with_disjoint_synthetic_ids_passes() -> None:
    examples = _load_fixture()
    validate_gold_set(examples, expected_size=5, min_spans_per_type=1, synthetic_ids=["tpl:000042"])


def test_example_with_surface_mismatch_raises_ner_validation_error() -> None:
    examples = _load_fixture()
    broken = NerExample(
        id="gold:9999",
        text="order ORD-99321 shipped yesterday",
        entities=[CharSpan(6, 15, "ORDER_ID", "WRONG")],
        source="gold",
        split="gold",
    )
    tampered = [*examples[:4], broken]
    with pytest.raises(NerValidationError, match="does not match"):
        validate_gold_set(tampered, expected_size=5, min_spans_per_type=1)


def test_span_counts_by_label_are_accurate() -> None:
    report = validate_gold_set(_load_fixture(), expected_size=5, min_spans_per_type=1)
    assert report.span_counts_by_label == {
        "ORDER_ID": 1,
        "PRODUCT": 1,
        "DATE": 2,
        "AMOUNT": 1,
        "ACCOUNT_REF": 2,
    }


def test_warnings_flag_labels_below_the_floor() -> None:
    report = validate_gold_set(_load_fixture(), expected_size=5, min_spans_per_type=2)
    warned_labels = {w.split(":")[0] for w in report.warnings}
    # ORDER_ID (1), PRODUCT (1) and AMOUNT (1) are below a floor of 2;
    # DATE (2) and ACCOUNT_REF (2) are not.
    assert warned_labels == {"ORDER_ID", "PRODUCT", "AMOUNT"}


def test_no_warnings_when_every_label_meets_the_floor() -> None:
    report = validate_gold_set(_load_fixture(), expected_size=5, min_spans_per_type=1)
    assert report.warnings == []


def test_default_thresholds_are_the_spec_values() -> None:
    from ml.data.ner.gold import EXPECTED_SIZE, MIN_SPANS_PER_TYPE

    assert EXPECTED_SIZE == 200
    assert MIN_SPANS_PER_TYPE == 15


@dataclass
class _FakeEnt:
    start_char: int
    end_char: int
    label_: str
    text: str


@dataclass
class _FakeDoc:
    ents: list[_FakeEnt]


def _fake_nlp_finding(label_: str, surface: str):
    def nlp(text: str) -> _FakeDoc:
        if surface not in text:
            return _FakeDoc([])
        start = text.index(surface)
        return _FakeDoc([_FakeEnt(start, start + len(surface), label_, surface)])

    return nlp


class TestProposeEntities:
    def test_rules_only_when_no_nlp_given(self) -> None:
        spans = propose_entities("order ORD-99321 shipped yesterday", nlp=None)
        assert {(s.label, s.text) for s in spans} == {
            ("ORDER_ID", "ORD-99321"),
            ("DATE", "yesterday"),
        }

    def test_mapped_spacy_label_is_added(self) -> None:
        nlp = _fake_nlp_finding("DATE", "March 3")
        spans = propose_entities("delivered on March 3 as promised", nlp=nlp)
        assert ("DATE", "March 3") in {(s.label, s.text) for s in spans}

    def test_unmapped_spacy_label_is_ignored(self) -> None:
        nlp = _fake_nlp_finding("PERSON", "March")
        spans = propose_entities("hello March, how are you", nlp=nlp)
        assert spans == []

    def test_spacy_money_maps_to_amount(self) -> None:
        nlp = _fake_nlp_finding("MONEY", "fifty bucks")
        spans = propose_entities("it cost me fifty bucks total", nlp=nlp)
        assert ("AMOUNT", "fifty bucks") in {(s.label, s.text) for s in spans}

    def test_overlapping_rules_and_spacy_proposals_resolve_to_one_span(self) -> None:
        # Rules already finds "$49.99" as AMOUNT; a fake spaCy MONEY hit on
        # the same text must not produce two overlapping candidate spans.
        nlp = _fake_nlp_finding("MONEY", "$49.99")
        spans = propose_entities("charged $49.99 today", nlp=nlp)
        overlapping = [s for s in spans if s.label == "AMOUNT"]
        assert len(overlapping) == 1

    def test_every_proposed_span_offset_is_correct(self) -> None:
        nlp = _fake_nlp_finding("DATE", "March 3")
        text = "order ORD-1 delivered on March 3"
        for span in propose_entities(text, nlp=nlp):
            assert text[span.start : span.end] == span.text


def _candidate(msg_id: str, text: str) -> GoldCandidate:
    return GoldCandidate(id=msg_id, text=text, proposed_entities=propose_entities(text))


_POOL = [
    _candidate("m1", "order ORD-99321 shipped yesterday"),
    _candidate("m2", "charged $49.99 for my iPhone 12 Pro Max"),
    _candidate("m3", "account 4455-9911 was debited twice"),
    _candidate("m4", "no entities in this one at all"),
    _candidate("m5", "case CASE99 open since last Tuesday"),
    _candidate("m6", "another message with nothing in it"),
    _candidate("m7", "thanks for your help today"),
    _candidate("m8", "please refund $10 for order ORD-1234"),
]


class TestSelectGoldCandidates:
    def test_returns_exactly_total(self) -> None:
        selection = select_gold_candidates(
            _POOL, random.Random(42), total=6, blind_count=2, target_spans_per_type=1
        )
        assert len(selection.candidates) == 6

    def test_blind_ids_subset_of_selected_with_exact_count(self) -> None:
        selection = select_gold_candidates(
            _POOL, random.Random(42), total=6, blind_count=2, target_spans_per_type=1
        )
        selected_ids = {c.id for c in selection.candidates}
        assert selection.blind_ids <= selected_ids
        assert len(selection.blind_ids) == 2

    def test_deterministic_given_seed(self) -> None:
        first = select_gold_candidates(_POOL, random.Random(42), total=6, blind_count=2)
        second = select_gold_candidates(_POOL, random.Random(42), total=6, blind_count=2)
        assert [c.id for c in first.candidates] == [c.id for c in second.candidates]
        assert first.blind_ids == second.blind_ids

    def test_does_not_hang_when_a_label_is_entirely_absent_from_the_pool(self) -> None:
        # None of _POOL's candidates carry PRODUCT except m2 -- target far
        # above what's available must not infinite-loop.
        selection = select_gold_candidates(
            _POOL, random.Random(42), total=8, blind_count=0, target_spans_per_type=100
        )
        assert len(selection.candidates) == 8

    def test_never_selects_more_than_the_pool_size(self) -> None:
        selection = select_gold_candidates(_POOL, random.Random(42), total=1000, blind_count=0)
        assert len(selection.candidates) == len(_POOL)


class TestRenderParseGoldMarkdown:
    def test_round_trip_preserves_ids_and_blind_flags(self) -> None:
        selection = select_gold_candidates(
            _POOL, random.Random(1), total=5, blind_count=2, target_spans_per_type=1
        )
        markdown = render_gold_markdown(selection)
        entries = parse_gold_markdown(markdown)

        assert [e[0] for e in entries] == [c.id for c in selection.candidates]
        assert {e[0] for e in entries if e[2]} == selection.blind_ids

    def test_blind_body_has_no_brackets(self) -> None:
        selection = select_gold_candidates(
            _POOL, random.Random(1), total=5, blind_count=2, target_spans_per_type=1
        )
        markdown = render_gold_markdown(selection)
        for _candidate_id, body, is_blind in parse_gold_markdown(markdown):
            if is_blind:
                assert "[" not in body
                assert "|" not in body


class TestImportGoldMarkdown:
    def test_valid_markdown_imports_correctly(self) -> None:
        markdown = (
            "## msg:m1\n"
            "order [ORD-99321|ORDER_ID] shipped [yesterday|DATE]\n\n"
            "## msg:m4  [blind]\n"
            "no entities in this one at all\n"
        )
        originals = {
            "m1": "order ORD-99321 shipped yesterday",
            "m4": "no entities in this one at all",
        }

        examples = import_gold_markdown(markdown, originals)

        assert [e.id for e in examples] == ["gold:m1", "gold:m4"]
        assert examples[0].entities == [
            CharSpan(6, 15, "ORDER_ID", "ORD-99321"),
            CharSpan(24, 33, "DATE", "yesterday"),
        ]
        assert examples[1].entities == []
        assert all(e.source == "gold" for e in examples)
        assert all(e.split == "gold" for e in examples)

    def test_edited_prose_that_no_longer_matches_original_is_rejected(self) -> None:
        markdown = "## msg:m1\norder [ORD-00000|ORDER_ID] shipped yesterday\n"
        originals = {"m1": "order ORD-99321 shipped yesterday"}
        with pytest.raises(GoldImportError, match="does not match the exported original"):
            import_gold_markdown(markdown, originals)

    def test_missing_manifest_entry_is_rejected(self) -> None:
        markdown = "## msg:unknown\nsome text\n"
        with pytest.raises(GoldImportError, match="not found in the export manifest"):
            import_gold_markdown(markdown, {})

    def test_malformed_markup_is_rejected(self) -> None:
        markdown = "## msg:m1\norder [ORD-99321 ORDER_ID] shipped yesterday\n"
        originals = {"m1": "order [ORD-99321 ORDER_ID] shipped yesterday"}
        with pytest.raises(GoldImportError, match="malformed markup"):
            import_gold_markdown(markdown, originals)

    def test_duplicate_heading_is_rejected(self) -> None:
        markdown = "## msg:m1\nhello world\n\n## msg:m1\nhello world again\n"
        originals = {"m1": "hello world"}
        with pytest.raises(GoldImportError, match="duplicate heading"):
            import_gold_markdown(markdown, originals)

    def test_full_export_import_round_trip_passes_validate_gold_set(self) -> None:
        selection = select_gold_candidates(
            _POOL, random.Random(42), total=6, blind_count=2, target_spans_per_type=1
        )
        markdown = render_gold_markdown(selection)
        originals = {c.id: c.text for c in selection.candidates}

        examples = import_gold_markdown(markdown, originals)

        report = validate_gold_set(
            examples, expected_size=6, min_spans_per_type=0, synthetic_ids=["tpl:000001"]
        )
        assert report.n_examples == 6
