import random

from ml.data.ner.schema import CharSpan
from ml.data.ner.shells import (
    is_fixed_point,
    partition_message_ids,
    sentence_splice,
    single_entity_span,
    slot_substitute,
    wrap,
    zero_entity_shell,
)
from ml.inference.base import EntitySpan


class TestPartitionMessageIds:
    def test_shell_and_gold_ids_are_disjoint(self) -> None:
        ids = [f"msg-{i}" for i in range(500)]
        partition = partition_message_ids(ids, seed=42)
        assert partition.shell_ids.isdisjoint(partition.gold_ids)

    def test_every_input_id_lands_in_exactly_one_bucket(self) -> None:
        ids = [f"msg-{i}" for i in range(500)]
        partition = partition_message_ids(ids, seed=42)
        assert partition.shell_ids | partition.gold_ids == set(ids)

    def test_gold_fraction_is_approximately_respected(self) -> None:
        ids = [f"msg-{i}" for i in range(1000)]
        partition = partition_message_ids(ids, seed=42, gold_fraction=0.10)
        assert 90 <= len(partition.gold_ids) <= 110

    def test_deterministic_given_seed(self) -> None:
        ids = [f"msg-{i}" for i in range(200)]
        first = partition_message_ids(ids, seed=42)
        second = partition_message_ids(ids, seed=42)
        assert first == second

    def test_different_seeds_produce_different_partitions(self) -> None:
        ids = [f"msg-{i}" for i in range(200)]
        first = partition_message_ids(ids, seed=1)
        second = partition_message_ids(ids, seed=2)
        assert first.gold_ids != second.gold_ids

    def test_stable_regardless_of_input_order(self) -> None:
        ids = [f"msg-{i}" for i in range(200)]
        shuffled = list(reversed(ids))
        assert partition_message_ids(ids, seed=42) == partition_message_ids(shuffled, seed=42)


class TestIsFixedPoint:
    def test_plain_text_is_fixed_point(self) -> None:
        assert is_fixed_point("order ORD-99321 shipped yesterday")

    def test_html_entity_is_not_fixed_point(self) -> None:
        assert not is_fixed_point("check &lt;this&gt; out")


class TestZeroEntityShell:
    def test_true_for_fixed_point_text_with_no_entities(self) -> None:
        assert zero_entity_shell("thanks for the help, really appreciated")

    def test_false_when_rules_baseline_finds_an_entity(self) -> None:
        assert not zero_entity_shell("charged $49.99 for this")

    def test_false_when_not_a_fixed_point(self) -> None:
        assert not zero_entity_shell("check &lt;this&gt; out")


class TestSingleEntitySpan:
    def test_returns_the_span_when_exactly_one_entity(self) -> None:
        span = single_entity_span("account 4455-9911 was debited")
        assert span is not None
        assert span.label == "ACCOUNT_REF"
        assert span.text == "4455-9911"

    def test_none_when_zero_entities(self) -> None:
        assert single_entity_span("thanks for the help") is None

    def test_none_when_multiple_entities(self) -> None:
        text = "order ORD-12345 charged $49.99"
        assert single_entity_span(text) is None

    def test_none_when_not_a_fixed_point(self) -> None:
        assert single_entity_span("check &lt;this&gt; out") is None


class TestWrap:
    def test_shell_first_shifts_rendered_spans(self) -> None:
        shell = "thanks for the help"
        rendered_text = "charged $40 today"
        rendered_spans = [CharSpan(8, 11, "AMOUNT", "$40")]
        text, spans = wrap(shell, rendered_text, rendered_spans, shell_first=True)
        assert text == "thanks for the help charged $40 today"
        assert spans == [CharSpan(28, 31, "AMOUNT", "$40")]
        assert text[spans[0].start : spans[0].end] == spans[0].text

    def test_shell_last_leaves_rendered_spans_unshifted(self) -> None:
        shell = "thanks for the help"
        rendered_text = "charged $40 today"
        rendered_spans = [CharSpan(8, 11, "AMOUNT", "$40")]
        text, spans = wrap(shell, rendered_text, rendered_spans, shell_first=False)
        assert text == "charged $40 today thanks for the help"
        assert spans == rendered_spans
        assert text[spans[0].start : spans[0].end] == spans[0].text


class TestSentenceSplice:
    def test_inserts_at_a_sentence_boundary(self) -> None:
        shell = "Thanks for the reply. Still waiting though."
        rendered_text = "charged $40 today"
        rendered_spans = [CharSpan(8, 11, "AMOUNT", "$40")]
        rng = random.Random(42)
        result = sentence_splice(shell, rendered_text, rendered_spans, rng)
        assert result is not None
        text, spans = result
        assert text[spans[0].start : spans[0].end] == spans[0].text
        assert "charged $40 today" in text

    def test_none_when_no_sentence_boundary(self) -> None:
        shell = "no punctuation here at all"
        rng = random.Random(42)
        result = sentence_splice(
            shell, "charged $40 today", [CharSpan(8, 11, "AMOUNT", "$40")], rng
        )
        assert result is None

    def test_result_is_a_clean_text_fixed_point(self) -> None:
        # Regression test: the naive version spliced at match.start() --
        # the position where the boundary's whitespace *begins* -- which
        # dropped the space before the inserted clause (shell ended
        # "help." directly against the clause with no gap) and then
        # doubled it after, since the shell's own original whitespace
        # (untouched, from match.start() onward) was concatenated right
        # after this function's own literal ". ". Both defects made the
        # result fail the clean_text fixed-point invariant every M4
        # component depends on.
        shell = "Thanks for the help. Really appreciated it a lot."
        rendered_text = "charged $40 today"
        rendered_spans = [CharSpan(8, 11, "AMOUNT", "$40")]
        rng = random.Random(0)

        result = sentence_splice(shell, rendered_text, rendered_spans, rng)

        assert result is not None
        text, spans = result
        assert is_fixed_point(text)
        assert ".charged" not in text
        assert ".  " not in text
        assert text[spans[0].start : spans[0].end] == spans[0].text

    def test_exactly_one_space_before_and_after_the_inserted_clause(self) -> None:
        shell = "Hello there. Goodbye now."
        rng = random.Random(1)
        result = sentence_splice(shell, "clause", [], rng)
        assert result is not None
        text, _ = result
        assert " clause. " in text


class TestSlotSubstitute:
    def test_replaces_target_span_and_offsets_correctly(self) -> None:
        shell = "account 4455-9911 was debited"
        target = EntitySpan(start=8, end=17, label="ACCOUNT_REF", text="4455-9911", score=1.0)
        text, spans = slot_substitute(shell, target, "ACC-000001")
        assert text == "account ACC-000001 was debited"
        assert spans == [CharSpan(8, 18, "ACCOUNT_REF", "ACC-000001")]
        assert text[spans[0].start : spans[0].end] == spans[0].text

    def test_handles_shorter_replacement_value(self) -> None:
        shell = "account 4455-9911 was debited"
        target = EntitySpan(start=8, end=17, label="ACCOUNT_REF", text="4455-9911", score=1.0)
        text, spans = slot_substitute(shell, target, "A1")
        assert text == "account A1 was debited"
        assert text[spans[0].start : spans[0].end] == spans[0].text
