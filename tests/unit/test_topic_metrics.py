import pytest

from ml.evaluation.topic_metrics import compute_topic_coherence

# 8 tiny hand-built "documents" (already tokenized). "refund"/"order" always
# appear together; "battery" and "delivery" never co-occur at all -- the two
# extremes NPMI is supposed to tell apart.
DOCUMENTS = [
    ["refund", "order", "late", "delivery"],
    ["refund", "order", "late"],
    ["refund", "order"],
    ["refund", "order", "package"],
    ["battery", "phone", "charge"],
    ["battery", "phone"],
    ["battery", "charge"],
    ["refund", "order", "delivery"],
]


def test_perfectly_co_occurring_terms_score_npmi_near_one() -> None:
    metrics = compute_topic_coherence({1: ["refund", "order"]}, DOCUMENTS)

    assert metrics.per_topic[0].npmi == pytest.approx(1.0, abs=1e-6)


def test_never_co_occurring_terms_score_npmi_near_negative_one() -> None:
    metrics = compute_topic_coherence({1: ["battery", "delivery"]}, DOCUMENTS)

    assert metrics.per_topic[0].npmi < -0.8


def test_coherent_topic_outranks_incoherent_topic() -> None:
    metrics = compute_topic_coherence(
        {1: ["refund", "order"], 2: ["battery", "delivery"]}, DOCUMENTS
    )

    per_topic = {t.topic_id: t.npmi for t in metrics.per_topic}
    assert per_topic[1] > per_topic[2]
    assert metrics.mean_npmi == pytest.approx((per_topic[1] + per_topic[2]) / 2)


def test_single_term_topic_scores_zero_but_still_counts() -> None:
    metrics = compute_topic_coherence({1: ["refund"], 2: ["refund", "order"]}, DOCUMENTS)

    per_topic = {t.topic_id: t.npmi for t in metrics.per_topic}
    assert per_topic[1] == 0.0
    assert metrics.n_topics == 2


def test_only_top_n_terms_are_used() -> None:
    # An 11th term that would tank coherence is ignored because TOP_N_TERMS
    # caps how many of a topic's ranked terms actually get scored.
    with_extra = compute_topic_coherence(
        {1: ["refund", "order", "late", "package", "delivery", "x", "y", "z", "q", "w", "battery"]},
        DOCUMENTS,
    )
    without_extra = compute_topic_coherence(
        {1: ["refund", "order", "late", "package", "delivery", "x", "y", "z", "q", "w"]},
        DOCUMENTS,
    )

    assert with_extra.per_topic[0].npmi == pytest.approx(without_extra.per_topic[0].npmi)


def test_to_metrics_dict_shape() -> None:
    metrics = compute_topic_coherence({1: ["refund", "order"]}, DOCUMENTS)

    assert set(metrics.to_metrics_dict()) == {"mean_npmi", "per_topic", "n_topics"}
    assert metrics.to_metrics_dict()["per_topic"] == [{"topic_id": 1, "npmi": pytest.approx(1.0)}]


def test_empty_documents_raises() -> None:
    with pytest.raises(ValueError, match="at least one document"):
        compute_topic_coherence({1: ["refund", "order"]}, [])
