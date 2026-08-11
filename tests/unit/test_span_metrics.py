import json

import pytest

from ml.data.ner.schema import CharSpan
from ml.evaluation.span_metrics import SpanMetrics, compute_span_metrics, span_errors

LABELS = ["ORDER_ID", "PRODUCT", "DATE", "AMOUNT", "ACCOUNT_REF"]


def test_perfect_predictions_score_one() -> None:
    gold = [[CharSpan(6, 15, "ORDER_ID", "ORD-99321")]]
    pred = [[CharSpan(6, 15, "ORDER_ID", "ORD-99321")]]

    metrics = compute_span_metrics(gold, pred, LABELS)

    assert metrics.micro_f1 == 1.0
    assert metrics.per_type["ORDER_ID"].f1 == 1.0
    assert metrics.per_type["ORDER_ID"].tp == 1
    assert metrics.per_type["ORDER_ID"].fp == 0
    assert metrics.per_type["ORDER_ID"].fn == 0


def test_completely_disjoint_predictions_score_zero() -> None:
    gold = [[CharSpan(0, 5, "ORDER_ID", "abcde")]]
    pred = [[CharSpan(10, 15, "PRODUCT", "fghij")]]

    metrics = compute_span_metrics(gold, pred, LABELS)

    assert metrics.micro_f1 == 0.0
    assert metrics.per_type["ORDER_ID"].fn == 1
    assert metrics.per_type["PRODUCT"].fp == 1


def test_off_by_one_boundary_counts_as_fp_and_fn_under_exact_match() -> None:
    gold = [[CharSpan(8, 14, "AMOUNT", "$49.99")]]
    pred = [[CharSpan(8, 13, "AMOUNT", "$49.9")]]

    metrics = compute_span_metrics(gold, pred, LABELS)

    assert metrics.per_type["AMOUNT"].tp == 0
    assert metrics.per_type["AMOUNT"].fp == 1
    assert metrics.per_type["AMOUNT"].fn == 1
    assert metrics.micro_f1 == 0.0


def test_off_by_one_boundary_still_counts_under_partial_match() -> None:
    gold = [[CharSpan(8, 14, "AMOUNT", "$49.99")]]
    pred = [[CharSpan(8, 13, "AMOUNT", "$49.9")]]

    metrics = compute_span_metrics(gold, pred, LABELS)

    # Overlapping ranges, same label -> partial counts it as a match even
    # though exact match (and therefore micro_f1/per_type) does not.
    assert metrics.partial_f1 == 1.0


def test_off_by_one_boundary_fails_boundary_match_when_label_also_differs() -> None:
    gold = [[CharSpan(8, 14, "AMOUNT", "$49.99")]]
    pred = [[CharSpan(9, 14, "PRODUCT", "49.99")]]

    metrics = compute_span_metrics(gold, pred, LABELS)

    # Different (start, end) *and* different label -> misses boundary_f1 too.
    assert metrics.boundary_f1 == 0.0


def test_correct_boundary_wrong_label_hits_boundary_not_exact() -> None:
    gold = [[CharSpan(8, 14, "AMOUNT", "$49.99")]]
    pred = [[CharSpan(8, 14, "PRODUCT", "$49.99")]]

    metrics = compute_span_metrics(gold, pred, LABELS)

    assert metrics.micro_f1 == 0.0
    assert metrics.boundary_f1 == 1.0


def test_duplicate_identical_predictions_score_one_tp_and_one_fp() -> None:
    gold = [[CharSpan(6, 15, "ORDER_ID", "ORD-99321")]]
    pred = [[CharSpan(6, 15, "ORDER_ID", "ORD-99321"), CharSpan(6, 15, "ORDER_ID", "ORD-99321")]]

    metrics = compute_span_metrics(gold, pred, LABELS)

    assert metrics.per_type["ORDER_ID"].tp == 1
    assert metrics.per_type["ORDER_ID"].fp == 1
    assert metrics.per_type["ORDER_ID"].fn == 0


def test_empty_gold_and_empty_pred_is_zero_not_undefined() -> None:
    metrics = compute_span_metrics([[]], [[]], LABELS)

    for label in LABELS:
        m = metrics.per_type[label]
        assert (m.precision, m.recall, m.f1) == (0.0, 0.0, 0.0)
    assert metrics.micro_f1 == 0.0
    assert metrics.macro_f1 == 0.0


def test_support_equals_tp_plus_fn() -> None:
    gold = [
        [CharSpan(0, 5, "ORDER_ID", "abcde"), CharSpan(10, 15, "ORDER_ID", "fghij")],
    ]
    pred = [[CharSpan(0, 5, "ORDER_ID", "abcde")]]

    metrics = compute_span_metrics(gold, pred, LABELS)

    assert metrics.per_type["ORDER_ID"].support == 2
    assert metrics.per_type["ORDER_ID"].tp == 1
    assert metrics.per_type["ORDER_ID"].fn == 1


def test_n_documents_and_n_gold_spans() -> None:
    gold = [
        [CharSpan(0, 5, "ORDER_ID", "abcde")],
        [],
        [CharSpan(0, 3, "DATE", "abc"), CharSpan(5, 8, "AMOUNT", "def")],
    ]
    pred = [[], [], []]

    metrics = compute_span_metrics(gold, pred, LABELS)

    assert metrics.n_documents == 3
    assert metrics.n_gold_spans == 3


def test_macro_f1_averages_across_all_given_labels_including_zero_support() -> None:
    gold = [[CharSpan(0, 5, "ORDER_ID", "abcde")]]
    pred = [[CharSpan(0, 5, "ORDER_ID", "abcde")]]

    metrics = compute_span_metrics(gold, pred, LABELS)

    # 1.0 (ORDER_ID) + 0.0 for each of the other 4 labels, averaged over 5.
    assert metrics.macro_f1 == 1.0 / len(LABELS)


def test_raises_on_mismatched_document_counts() -> None:
    with pytest.raises(ValueError, match="same number of documents"):
        compute_span_metrics([[]], [[], []], LABELS)


def test_bootstrap_ci_bounds_are_ordered_and_within_zero_one() -> None:
    gold = [[CharSpan(0, 5, "ORDER_ID", "abcde")] for _ in range(20)]
    pred = [[CharSpan(0, 5, "ORDER_ID", "abcde")] for _ in range(20)]

    metrics = compute_span_metrics(gold, pred, LABELS, bootstrap_resamples=200)

    ci = metrics.per_type["ORDER_ID"]
    assert 0.0 <= ci.f1_ci_low <= ci.f1_ci_high <= 1.0


def test_bootstrap_is_deterministic_given_seed() -> None:
    gold = [[CharSpan(0, 5, "ORDER_ID", "abcde")], []]
    pred = [[], [CharSpan(0, 5, "ORDER_ID", "abcde")]]

    first = compute_span_metrics(gold, pred, LABELS, bootstrap_resamples=100, seed=7)
    second = compute_span_metrics(gold, pred, LABELS, bootstrap_resamples=100, seed=7)

    assert first.per_type["ORDER_ID"].f1_ci_low == second.per_type["ORDER_ID"].f1_ci_low
    assert first.per_type["ORDER_ID"].f1_ci_high == second.per_type["ORDER_ID"].f1_ci_high


def test_to_metrics_dict_is_json_serializable_and_round_trips_key_numbers() -> None:
    gold = [[CharSpan(6, 15, "ORDER_ID", "ORD-99321")]]
    pred = [[CharSpan(6, 15, "ORDER_ID", "ORD-99321")]]

    metrics = compute_span_metrics(gold, pred, LABELS)
    as_dict = metrics.to_metrics_dict()
    round_tripped = json.loads(json.dumps(as_dict))

    assert round_tripped["micro_f1"] == metrics.micro_f1
    assert round_tripped["per_type"]["ORDER_ID"]["f1"] == 1.0
    assert round_tripped["labels"] == LABELS
    assert round_tripped["n_documents"] == 1


def test_to_metrics_dict_return_type() -> None:
    metrics = compute_span_metrics([[]], [[]], LABELS)
    assert isinstance(metrics, SpanMetrics)
    assert set(metrics.to_metrics_dict()) == {
        "per_type",
        "micro_precision",
        "micro_recall",
        "micro_f1",
        "macro_f1",
        "boundary_f1",
        "partial_f1",
        "labels",
        "n_documents",
        "n_gold_spans",
    }


class TestSpanErrors:
    def test_false_positive_for_unmatched_prediction(self) -> None:
        gold = [[]]
        pred = [[CharSpan(0, 3, "PRODUCT", "abc")]]
        texts = ["abc def"]

        fps, fns = span_errors(gold, pred, texts)

        assert len(fps) == 1
        assert fps[0].document_index == 0
        assert fps[0].surface == "abc"
        assert fns == []

    def test_false_negative_for_unmatched_gold(self) -> None:
        gold = [[CharSpan(0, 3, "PRODUCT", "abc")]]
        pred = [[]]
        texts = ["abc def"]

        fps, fns = span_errors(gold, pred, texts)

        assert fps == []
        assert len(fns) == 1
        assert fns[0].surface == "abc"

    def test_matched_spans_produce_no_errors(self) -> None:
        gold = [[CharSpan(0, 3, "PRODUCT", "abc")]]
        pred = [[CharSpan(0, 3, "PRODUCT", "abc")]]
        texts = ["abc def"]

        fps, fns = span_errors(gold, pred, texts)

        assert fps == []
        assert fns == []

    def test_duplicate_gold_with_single_prediction_yields_one_false_negative(self) -> None:
        gold = [[CharSpan(0, 3, "PRODUCT", "abc"), CharSpan(0, 3, "PRODUCT", "abc")]]
        pred = [[CharSpan(0, 3, "PRODUCT", "abc")]]
        texts = ["abc def"]

        fps, fns = span_errors(gold, pred, texts)

        assert fps == []
        assert len(fns) == 1

    def test_raises_on_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            span_errors([[]], [[]], [])
