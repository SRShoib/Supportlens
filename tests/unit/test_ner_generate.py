import random

from ml.data.ner.generate import _classify_shells, compose_dataset
from ml.data.ner.schema import validate_example
from ml.inference.rules_ner import extract_spans

_ZERO_ENTITY_SHELLS = [
    (f"m{i}", text)
    for i, text in enumerate(
        [
            "thanks for the reply, still no update though",
            "this has been so frustrating honestly",
            "hope you can help me out here",
            "not sure what happened but it's not working",
            "checking in again on this issue",
            "any news on my case yet?",
            "still no response from anyone here",
            "please look into this when you get a chance",
        ]
    )
]
_SINGLE_ENTITY_SHELLS = [
    ("s1", "account 4455-9911 was debited twice"),
    ("s2", "charged $19.99 for something I don't recognize"),
]
_NOT_FIXED_POINT_SHELLS = [("bad1", "check &lt;this&gt; out")]

_MIXED_SHELL_POOL = _ZERO_ENTITY_SHELLS + _SINGLE_ENTITY_SHELLS + _NOT_FIXED_POINT_SHELLS


class TestClassifyShells:
    def test_splits_into_zero_and_single_entity_buckets(self) -> None:
        zero_entity, single_entity, _ = _classify_shells(_MIXED_SHELL_POOL)
        assert {msg_id for msg_id, _ in zero_entity} == {m for m, _ in _ZERO_ENTITY_SHELLS}
        assert {msg_id for msg_id, _ in single_entity} == {m for m, _ in _SINGLE_ENTITY_SHELLS}

    def test_counts_non_fixed_point_shells_as_dropped(self) -> None:
        _, _, dropped = _classify_shells(_MIXED_SHELL_POOL)
        assert dropped == len(_NOT_FIXED_POINT_SHELLS)

    def test_multi_entity_shell_lands_in_neither_bucket_without_inflating_drops(self) -> None:
        multi = [("multi", "order ORD-12345 charged $49.99")]
        zero_entity, single_entity, dropped = _classify_shells(multi)
        assert zero_entity == []
        assert single_entity == []
        assert dropped == 0


class TestComposeDataset:
    def test_produces_requested_total(self) -> None:
        examples, stats = compose_dataset(random.Random(42), _MIXED_SHELL_POOL, n_total=200)
        assert len(examples) == 200
        assert stats.n_total == 200

    def test_every_example_passes_validate_example(self) -> None:
        examples, _ = compose_dataset(random.Random(42), _MIXED_SHELL_POOL, n_total=200)
        for example in examples:
            validate_example(example)  # raises on any invariant violation

    def test_same_seed_produces_byte_identical_output(self) -> None:
        first, _ = compose_dataset(random.Random(42), _MIXED_SHELL_POOL, n_total=200)
        second, _ = compose_dataset(random.Random(42), _MIXED_SHELL_POOL, n_total=200)
        assert first == second

    def test_different_seed_produces_different_output(self) -> None:
        first, _ = compose_dataset(random.Random(1), _MIXED_SHELL_POOL, n_total=200)
        second, _ = compose_dataset(random.Random(2), _MIXED_SHELL_POOL, n_total=200)
        assert first != second

    def test_not_fixed_point_shell_increments_drop_counter_not_raises(self) -> None:
        # compose_dataset must not raise just because the shell pool passed
        # to it contains an unusable message -- it counts the drop and
        # proceeds with what's usable.
        _, stats = compose_dataset(random.Random(42), _MIXED_SHELL_POOL, n_total=200)
        assert stats.shells_dropped_not_fixed_point == len(_NOT_FIXED_POINT_SHELLS)

    def test_negative_examples_present_and_have_no_entities(self) -> None:
        examples, stats = compose_dataset(random.Random(42), _MIXED_SHELL_POOL, n_total=200)
        negatives = [e for e in examples if e.source == "negative"]
        assert negatives, "expected at least one negative example"
        assert stats.by_source.get("negative", 0) == len(negatives)
        for example in negatives:
            assert example.entities == []

    def test_every_source_bucket_present_given_a_rich_enough_shell_pool(self) -> None:
        _, stats = compose_dataset(random.Random(42), _MIXED_SHELL_POOL, n_total=200)
        assert set(stats.by_source) == {
            "template",
            "shell_wrap",
            "shell_sentence",
            "shell_slot_sub",
            "negative",
        }

    def test_slot_sub_count_is_capped_by_available_single_entity_shells(self) -> None:
        _, stats = compose_dataset(random.Random(42), _MIXED_SHELL_POOL, n_total=4000)
        assert stats.by_source["shell_slot_sub"] <= len(_SINGLE_ENTITY_SHELLS)

    def test_no_shell_backed_example_references_a_message_id_outside_the_pool(self) -> None:
        pool_ids = {msg_id for msg_id, _ in _MIXED_SHELL_POOL}
        examples, _ = compose_dataset(random.Random(42), _MIXED_SHELL_POOL, n_total=200)
        for example in examples:
            if example.source in ("shell_wrap", "shell_sentence", "shell_slot_sub", "negative"):
                referenced_id = example.id.split(":")[1]
                assert referenced_id in pool_ids

    def test_splits_are_roughly_70_15_15(self) -> None:
        examples, _ = compose_dataset(random.Random(42), _MIXED_SHELL_POOL, n_total=2000)
        counts = {"train": 0, "val": 0, "test": 0}
        for example in examples:
            counts[example.split] += 1
        assert 1300 <= counts["train"] <= 1500
        assert 200 <= counts["val"] <= 400
        assert 200 <= counts["test"] <= 400

    def test_falls_back_gracefully_with_no_usable_shells_at_all(self) -> None:
        examples, stats = compose_dataset(random.Random(42), [], n_total=200)
        # No shell pool at all -> only pure-template examples are possible.
        assert examples
        assert set(stats.by_source) == {"template"}
        for example in examples:
            validate_example(example)

    def test_rules_baseline_finds_no_entities_in_negative_examples(self) -> None:
        # Cross-check against the actual rules baseline, not just the
        # generator's own classification logic.
        examples, _ = compose_dataset(random.Random(42), _MIXED_SHELL_POOL, n_total=200)
        for example in examples:
            if example.source == "negative":
                assert extract_spans(example.text) == []
