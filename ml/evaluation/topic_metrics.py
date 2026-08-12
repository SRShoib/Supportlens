"""NPMI topic coherence (SPEC M7: "≥ 30 coherent topics" -- SPEC names no
coherence metric or threshold, so this module supplies the measurable
stand-in: normalized pointwise mutual information (NPMI) averaged over
every pair of a topic's top-N c-TF-IDF terms, the standard formulation from
Lau, Newman & Baldwin (2014), "Machine Reading Tea Leaves". Computed
in-repo from the corpus's own document co-occurrence counts -- no gensim
dependency, no reference corpus. See docs/decisions.md for why NPMI and
not a hardcoded pass/fail threshold.

Pure stdlib, no DB import -- same reasoning as trend_metrics.py's module
docstring: this needs to be testable and importable without the `topics`
dependency group installed.

NPMI ranges [-1, 1]: -1 means the pair never co-occurs, +1 means they
always co-occur together, 0 is chance-level co-occurrence. Zero-count
pairs are additive-smoothed by EPSILON rather than skipped, so a topic
whose terms genuinely never co-occur scores a real (strongly negative)
number instead of being silently dropped from the average.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

TOP_N_TERMS = 10
EPSILON = 1e-12


@dataclass(frozen=True)
class TopicCoherence:
    topic_id: int
    npmi: float


@dataclass(frozen=True)
class CoherenceMetrics:
    mean_npmi: float
    per_topic: list[TopicCoherence]
    n_topics: int

    def to_metrics_dict(self) -> dict[str, object]:
        return {
            "mean_npmi": self.mean_npmi,
            "per_topic": [{"topic_id": t.topic_id, "npmi": t.npmi} for t in self.per_topic],
            "n_topics": self.n_topics,
        }


def _npmi(doc_freq_a: int, doc_freq_b: int, co_doc_freq: int, n_docs: int) -> float:
    p_a = max(doc_freq_a / n_docs, EPSILON)
    p_b = max(doc_freq_b / n_docs, EPSILON)
    p_ab = max(co_doc_freq / n_docs, EPSILON)
    denom = -math.log(p_ab)
    if denom == 0:
        # p_ab == 1: the pair co-occurs in literally every document.
        return 1.0
    return math.log(p_ab / (p_a * p_b)) / denom


def compute_topic_coherence(
    topic_terms: dict[int, Sequence[str]], documents: Sequence[Sequence[str]]
) -> CoherenceMetrics:
    """topic_terms: topic_id -> its top-N c-TF-IDF terms, already ranked
    (only the first TOP_N_TERMS are used). documents: tokenized documents
    from the SAME corpus the topic model was fit on -- word presence per
    document, order and repeats within a document don't matter here.

    A topic with fewer than 2 terms scores npmi=0.0 (no pair to score)
    rather than being excluded from n_topics/mean_npmi.
    """
    n_docs = len(documents)
    if n_docs == 0:
        raise ValueError("compute_topic_coherence requires at least one document")

    doc_sets = [set(doc) for doc in documents]
    vocab = {term for terms in topic_terms.values() for term in terms[:TOP_N_TERMS]}
    doc_freq = {word: sum(1 for doc in doc_sets if word in doc) for word in vocab}

    per_topic: list[TopicCoherence] = []
    for topic_id in sorted(topic_terms):
        top_terms = list(topic_terms[topic_id])[:TOP_N_TERMS]
        pairs = list(combinations(top_terms, 2))
        if not pairs:
            per_topic.append(TopicCoherence(topic_id, 0.0))
            continue

        pair_scores = []
        for word_a, word_b in pairs:
            co_doc_freq = sum(1 for doc in doc_sets if word_a in doc and word_b in doc)
            pair_scores.append(_npmi(doc_freq[word_a], doc_freq[word_b], co_doc_freq, n_docs))
        per_topic.append(TopicCoherence(topic_id, sum(pair_scores) / len(pair_scores)))

    mean_npmi = sum(t.npmi for t in per_topic) / len(per_topic) if per_topic else 0.0
    return CoherenceMetrics(mean_npmi=mean_npmi, per_topic=per_topic, n_topics=len(per_topic))
