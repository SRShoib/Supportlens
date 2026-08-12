from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.training.topic_model import (
    TopicModelConfig,
    TopicRecord,
    _ctfidf_keywords,
    _label_from_keywords,
    export_variant,
    fit_kmeans_baseline,
)

# fit_bertopic() is deliberately untested here: it needs the `topics`
# dependency group (bertopic/umap/hdbscan), which isn't installed by
# default or in CI -- see this module's docstring and pyproject.toml. The
# functions below (config parsing, c-TF-IDF labeling, the KMeans baseline,
# artifact export) only need numpy/pandas/sklearn, all default deps, so
# they're fully unit-testable.


def test_from_yaml_parses_defaults_and_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / "topics.yaml"
    config_path.write_text("kmeans_n_clusters: 12\numap_min_dist: 0.0\nseed: 7\n", encoding="utf-8")

    config = TopicModelConfig.from_yaml(config_path)

    assert config.kmeans_n_clusters == 12
    assert config.umap_min_dist == 0.0
    assert isinstance(config.umap_min_dist, float)
    assert config.seed == 7
    assert config.hdbscan_min_cluster_size == 75  # untouched default


def test_label_from_keywords_empty_is_outliers() -> None:
    assert _label_from_keywords([]) == "outliers"


def test_label_from_keywords_joins_top_n() -> None:
    assert _label_from_keywords(["refund", "order", "late", "delivery", "package"]) == (
        "refund, order, late, delivery"
    )


def test_ctfidf_keywords_distinguishes_two_clusters() -> None:
    documents = [
        "refund my order please",
        "order refund is late",
        "battery drains too fast",
        "phone battery is bad",
    ]
    assignments = [0, 0, 1, 1]

    topic_0_keywords = _ctfidf_keywords(documents, assignments, topic_key=0)
    topic_1_keywords = _ctfidf_keywords(documents, assignments, topic_key=1)

    assert "refund" in topic_0_keywords or "order" in topic_0_keywords
    assert "battery" in topic_1_keywords
    assert "battery" not in topic_0_keywords
    assert "refund" not in topic_1_keywords


def test_ctfidf_keywords_empty_cluster_returns_empty_list() -> None:
    assert _ctfidf_keywords(["a"], [0], topic_key=99) == []


def test_ctfidf_keywords_filters_masking_tokens_and_english_stopwords() -> None:
    # Regression test: ml/data/masking.py's <USER>/<URL> tokens are near-
    # universal (almost every real ticket mentions the brand's @handle or a
    # URL) -- a plain TfidfVectorizer's default tokenizer strips the angle
    # brackets before counting, collapsing them into bare "user"/"url" that
    # then dominate every topic's keywords instead of being suppressed the
    # way a true common-to-every-cluster term should be (found on the real
    # M7 run: "user, url, the, in" as a top-ranked topic label).
    documents = [
        "<USER> <URL> refund my order please",
        "<USER> <URL> order refund is late",
        "<USER> <URL> please refund this order",
        "<USER> <URL> battery drains too fast",
        "<USER> <URL> phone battery is bad",
        "<USER> <URL> battery charge broken",
    ]
    assignments = [0, 0, 0, 1, 1, 1]

    keywords = _ctfidf_keywords(documents, assignments, topic_key=0)

    assert "user" not in keywords
    assert "url" not in keywords
    assert "the" not in keywords  # plain English stopword, not just a mask artifact
    assert "refund" in keywords


def test_fit_kmeans_baseline_separates_two_obvious_clusters() -> None:
    embeddings = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [10.0, 10.0],
            [10.1, 10.1],
            [10.2, 10.0],
        ]
    )
    documents = [
        "refund my order please",
        "order refund is late",
        "please refund this order",
        "battery drains too fast",
        "phone battery is bad",
        "battery charge broken",
    ]
    config = TopicModelConfig(kmeans_n_clusters=2, seed=42)

    assignments, records = fit_kmeans_baseline(embeddings, documents, config)

    assert len(assignments) == 6
    assert assignments[0] == assignments[1] == assignments[2]
    assert assignments[3] == assignments[4] == assignments[5]
    assert assignments[0] != assignments[3]
    assert {r.topic_key for r in records} == set(assignments)
    for record in records:
        assert record.size == 3
        assert record.keywords  # every cluster gets a non-empty label here


def test_export_variant_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import ml.training.topic_model as topic_model_module

    monkeypatch.setattr(topic_model_module, "MODELS_DIR", tmp_path)
    records = [
        TopicRecord(topic_key=-1, label="outliers", keywords=[], size=2),
        TopicRecord(topic_key=0, label="refund, order", keywords=["refund", "order"], size=3),
    ]

    out_dir = export_variant("kmeans", ["t1", "t2", "t3"], [0, 0, -1], records)

    assert out_dir == tmp_path / "topics_kmeans_v1"
    topics_json = (out_dir / "topics.json").read_text(encoding="utf-8")
    assert '"model_version": "topics_kmeans_v1"' in topics_json
    assert '"topic_key": 0' in topics_json

    assignments = pd.read_parquet(out_dir / "assignments.parquet")
    assert assignments["ticket_id"].tolist() == ["t1", "t2", "t3"]
    assert assignments["topic_key"].tolist() == [0, 0, -1]
    assert assignments["probability"].tolist() == [1.0, 1.0, 1.0]
