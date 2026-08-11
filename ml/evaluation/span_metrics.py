"""Span-level P/R/F1 for M4 (SPEC: "span-level F1 reported per entity on the
gold set"). Deliberately NOT seqeval: seqeval scores BIO tag sequences,
which would force converting gold char spans -> tokens -> BIO, injecting a
tokenizer into the metric. The rules baseline (ml/inference/rules_ner.py)
has no tokenizer, so scoring it through a tokenizer-dependent metric would
make it incomparable to the transformer on the exact number M4 exists to
report. A char-span set metric is scheme-agnostic: anything exposing
start/end/label -- rules output, model output, or hand-annotated gold --
scores through the identical lens.

Exact match on (start, end, label) is the headline. boundary_f1 (label-blind)
and partial_f1 (overlap-relaxed, type-sensitive) are secondary diagnostics
that separate boundary errors from type-confusion errors.
"""

import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol


class SpanLike(Protocol):
    start: int
    end: int
    label: str


@dataclass(frozen=True)
class SpanTypeMetrics:
    label: str
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    support: int
    f1_ci_low: float
    f1_ci_high: float


@dataclass(frozen=True)
class SpanMetrics:
    per_type: dict[str, SpanTypeMetrics]
    micro_precision: float
    micro_recall: float
    micro_f1: float
    macro_f1: float
    boundary_f1: float
    partial_f1: float
    labels: list[str]
    n_documents: int
    n_gold_spans: int

    def to_metrics_dict(self) -> dict[str, Any]:
        return {
            "per_type": {
                label: {
                    "precision": m.precision,
                    "recall": m.recall,
                    "f1": m.f1,
                    "tp": m.tp,
                    "fp": m.fp,
                    "fn": m.fn,
                    "support": m.support,
                    "f1_ci_low": m.f1_ci_low,
                    "f1_ci_high": m.f1_ci_high,
                }
                for label, m in self.per_type.items()
            },
            "micro_precision": self.micro_precision,
            "micro_recall": self.micro_recall,
            "micro_f1": self.micro_f1,
            "macro_f1": self.macro_f1,
            "boundary_f1": self.boundary_f1,
            "partial_f1": self.partial_f1,
            "labels": self.labels,
            "n_documents": self.n_documents,
            "n_gold_spans": self.n_gold_spans,
        }


@dataclass(frozen=True)
class SpanError:
    document_index: int
    text: str
    start: int
    end: int
    label: str
    surface: str


def _span_key(span: SpanLike) -> tuple[int, int, str]:
    return (span.start, span.end, span.label)


def _boundary_key(span: SpanLike) -> tuple[int, int]:
    return (span.start, span.end)


def _prf1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _match_counts(gold_counter: Counter[Any], pred_counter: Counter[Any]) -> tuple[int, int, int]:
    """(tp, fp, fn) via multiset intersection -- Counter's `&` takes the
    element-wise min, so two identical duplicate predictions score one TP
    against one gold occurrence and one FP against the extra copy, rather
    than both scoring as TP."""
    tp = sum((gold_counter & pred_counter).values())
    fp = sum(pred_counter.values()) - tp
    fn = sum(gold_counter.values()) - tp
    return tp, fp, fn


def _bootstrap_f1_ci(
    per_doc_counts: Sequence[tuple[int, int, int]], resamples: int, seed: int
) -> tuple[float, float]:
    """Document-level percentile bootstrap (2.5%/97.5%), resampling whole
    documents with replacement -- not individual spans, since spans within
    one document aren't independent observations."""
    n = len(per_doc_counts)
    if n == 0 or resamples <= 0:
        return 0.0, 0.0

    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(resamples):
        tp = fp = fn = 0
        for _ in range(n):
            doc_tp, doc_fp, doc_fn = per_doc_counts[rng.randrange(n)]
            tp += doc_tp
            fp += doc_fp
            fn += doc_fn
        _, _, f1 = _prf1(tp, fp, fn)
        samples.append(f1)

    samples.sort()
    low = samples[int(0.025 * resamples)]
    high = samples[min(int(0.975 * resamples), resamples - 1)]
    return low, high


def _overlaps(a: SpanLike, b: SpanLike) -> bool:
    return a.start < b.end and b.start < a.end


def _compute_boundary_f1(
    gold: Sequence[Sequence[SpanLike]], pred: Sequence[Sequence[SpanLike]]
) -> float:
    total_tp = total_fp = total_fn = 0
    for gold_doc, pred_doc in zip(gold, pred, strict=True):
        gold_counter = Counter(_boundary_key(s) for s in gold_doc)
        pred_counter = Counter(_boundary_key(s) for s in pred_doc)
        tp, fp, fn = _match_counts(gold_counter, pred_counter)
        total_tp += tp
        total_fp += fp
        total_fn += fn
    _, _, f1 = _prf1(total_tp, total_fp, total_fn)
    return f1


def _compute_partial_f1(
    gold: Sequence[Sequence[SpanLike]], pred: Sequence[Sequence[SpanLike]]
) -> float:
    """Overlap-relaxed, type-sensitive: a predicted span matches a gold span
    of the same label if their ranges overlap at all. Greedy per-label
    matching within each document, first-available-overlap wins -- not
    optimal bipartite matching, but good enough for a diagnostic number."""
    total_tp = total_fp = total_fn = 0
    for gold_doc, pred_doc in zip(gold, pred, strict=True):
        gold_by_label: dict[str, list[SpanLike]] = {}
        pred_by_label: dict[str, list[SpanLike]] = {}
        for span in gold_doc:
            gold_by_label.setdefault(span.label, []).append(span)
        for span in pred_doc:
            pred_by_label.setdefault(span.label, []).append(span)

        for label in set(gold_by_label) | set(pred_by_label):
            gold_spans = gold_by_label.get(label, [])
            pred_spans = pred_by_label.get(label, [])
            matched_pred = [False] * len(pred_spans)
            tp = 0
            for gold_span in gold_spans:
                for i, pred_span in enumerate(pred_spans):
                    if not matched_pred[i] and _overlaps(gold_span, pred_span):
                        matched_pred[i] = True
                        tp += 1
                        break
            total_tp += tp
            total_fp += len(pred_spans) - sum(matched_pred)
            total_fn += len(gold_spans) - tp
    _, _, f1 = _prf1(total_tp, total_fp, total_fn)
    return f1


def compute_span_metrics(
    gold: Sequence[Sequence[SpanLike]],
    pred: Sequence[Sequence[SpanLike]],
    labels: Sequence[str],
    *,
    bootstrap_resamples: int = 1000,
    seed: int = 42,
) -> SpanMetrics:
    """gold and pred are paired, one entry per document (SPEC's gold set is
    per-message). Exact match on (start, end, label)."""
    if len(gold) != len(pred):
        raise ValueError("gold and pred must have the same number of documents (paired)")

    per_doc_label_counts: list[dict[str, tuple[int, int, int]]] = []
    for gold_doc, pred_doc in zip(gold, pred, strict=True):
        gold_counter = Counter(_span_key(s) for s in gold_doc)
        pred_counter = Counter(_span_key(s) for s in pred_doc)
        doc_counts: dict[str, tuple[int, int, int]] = {}
        for label in labels:
            g = Counter({k: v for k, v in gold_counter.items() if k[2] == label})
            p = Counter({k: v for k, v in pred_counter.items() if k[2] == label})
            doc_counts[label] = _match_counts(g, p)
        per_doc_label_counts.append(doc_counts)

    per_type: dict[str, SpanTypeMetrics] = {}
    for label in labels:
        tp = sum(doc[label][0] for doc in per_doc_label_counts)
        fp = sum(doc[label][1] for doc in per_doc_label_counts)
        fn = sum(doc[label][2] for doc in per_doc_label_counts)
        precision, recall, f1 = _prf1(tp, fp, fn)
        ci_low, ci_high = _bootstrap_f1_ci(
            [doc[label] for doc in per_doc_label_counts], bootstrap_resamples, seed
        )
        per_type[label] = SpanTypeMetrics(
            label=label,
            precision=precision,
            recall=recall,
            f1=f1,
            tp=tp,
            fp=fp,
            fn=fn,
            support=tp + fn,
            f1_ci_low=ci_low,
            f1_ci_high=ci_high,
        )

    micro_tp = sum(m.tp for m in per_type.values())
    micro_fp = sum(m.fp for m in per_type.values())
    micro_fn = sum(m.fn for m in per_type.values())
    micro_precision, micro_recall, micro_f1 = _prf1(micro_tp, micro_fp, micro_fn)
    macro_f1 = sum(m.f1 for m in per_type.values()) / len(labels) if labels else 0.0

    return SpanMetrics(
        per_type=per_type,
        micro_precision=micro_precision,
        micro_recall=micro_recall,
        micro_f1=micro_f1,
        macro_f1=macro_f1,
        boundary_f1=_compute_boundary_f1(gold, pred),
        partial_f1=_compute_partial_f1(gold, pred),
        labels=list(labels),
        n_documents=len(gold),
        n_gold_spans=sum(len(doc) for doc in gold),
    )


def span_errors(
    gold: Sequence[Sequence[SpanLike]],
    pred: Sequence[Sequence[SpanLike]],
    texts: Sequence[str],
) -> tuple[list[SpanError], list[SpanError]]:
    """(false_positives, false_negatives) under the same exact-match,
    multiset semantics as compute_span_metrics -- for the report's failure
    examples section, not for scoring."""
    if not (len(gold) == len(pred) == len(texts)):
        raise ValueError("gold, pred, and texts must be the same length (paired)")

    false_positives: list[SpanError] = []
    false_negatives: list[SpanError] = []

    for doc_index, (gold_doc, pred_doc, text) in enumerate(zip(gold, pred, texts, strict=True)):
        matched = Counter(_span_key(s) for s in gold_doc) & Counter(_span_key(s) for s in pred_doc)

        remaining = Counter(matched)
        for span in pred_doc:
            key = _span_key(span)
            if remaining[key] > 0:
                remaining[key] -= 1
                continue
            false_positives.append(
                SpanError(
                    doc_index, text, span.start, span.end, span.label, text[span.start : span.end]
                )
            )

        remaining = Counter(matched)
        for span in gold_doc:
            key = _span_key(span)
            if remaining[key] > 0:
                remaining[key] -= 1
                continue
            false_negatives.append(
                SpanError(
                    doc_index, text, span.start, span.end, span.label, text[span.start : span.end]
                )
            )

    return false_positives, false_negatives
