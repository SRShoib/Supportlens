from ml.evaluation.metrics import compute_classification_metrics


def test_perfect_predictions_score_macro_f1_one() -> None:
    y_true = ["a", "b", "c", "a", "b", "c"]
    y_pred = ["a", "b", "c", "a", "b", "c"]

    metrics = compute_classification_metrics(y_true, y_pred, labels=["a", "b", "c"])

    assert metrics.macro_f1 == 1.0
    assert all(f1 == 1.0 for f1 in metrics.per_class_f1.values())


def test_all_wrong_predictions_score_macro_f1_zero() -> None:
    y_true = ["a", "a", "a"]
    y_pred = ["b", "b", "b"]

    metrics = compute_classification_metrics(y_true, y_pred, labels=["a", "b"])

    assert metrics.macro_f1 == 0.0


def test_confusion_matrix_shape_matches_labels() -> None:
    y_true = ["a", "b", "a", "b"]
    y_pred = ["a", "a", "b", "b"]

    metrics = compute_classification_metrics(y_true, y_pred, labels=["a", "b"])

    assert len(metrics.confusion_matrix) == 2
    assert all(len(row) == 2 for row in metrics.confusion_matrix)
    assert sum(sum(row) for row in metrics.confusion_matrix) == 4


def test_per_class_f1_has_entry_for_every_label() -> None:
    metrics = compute_classification_metrics(["a", "b"], ["a", "a"], labels=["a", "b", "c"])

    assert set(metrics.per_class_f1) == {"a", "b", "c"}
    assert metrics.per_class_f1["c"] == 0.0  # never appears in true or pred


def test_labels_preserved_in_order() -> None:
    metrics = compute_classification_metrics(["x", "y"], ["x", "y"], labels=["y", "x"])
    assert metrics.labels == ["y", "x"]
