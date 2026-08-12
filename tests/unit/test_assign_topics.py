from pathlib import Path

import pytest

from ml.training.topic_model import TopicRecord, export_variant
from scripts.assign_topics import (
    TopicAssignment,
    TopicCatalogEntry,
    load_assignments,
    load_topic_catalog,
)


def test_load_topic_catalog_and_assignments_roundtrip_export_variant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ml.training.topic_model as topic_model_module

    monkeypatch.setattr(topic_model_module, "MODELS_DIR", tmp_path)
    records = [
        TopicRecord(topic_key=-1, label="outliers", keywords=[], size=1),
        TopicRecord(topic_key=0, label="refund, order", keywords=["refund", "order"], size=2),
    ]
    out_dir = export_variant("kmeans", ["t1", "t2", "t3"], [-1, 0, 0], records, [0.5, 0.9, 0.8])

    model_version, catalog = load_topic_catalog(out_dir)
    assignments = load_assignments(out_dir)

    assert model_version == "topics_kmeans_v1"
    assert catalog == [
        TopicCatalogEntry(topic_key=-1, label="outliers", keywords=[], size=1),
        TopicCatalogEntry(topic_key=0, label="refund, order", keywords=["refund", "order"], size=2),
    ]
    assert assignments == [
        TopicAssignment(ticket_id="t1", topic_key=-1, probability=0.5),
        TopicAssignment(ticket_id="t2", topic_key=0, probability=0.9),
        TopicAssignment(ticket_id="t3", topic_key=0, probability=0.8),
    ]
