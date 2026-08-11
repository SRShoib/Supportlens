from dataclasses import dataclass

from ml.training import tweet_eval_data


@dataclass
class _FakeLabelFeature:
    names: list[str]


class _FakeDataset:
    def __init__(self, rows: list[dict[str, object]], label_names: list[str]) -> None:
        self._rows = rows
        self.features = {"label": _FakeLabelFeature(names=label_names)}

    def __iter__(self):
        return iter(self._rows)

    def __len__(self) -> int:
        return len(self._rows)


class _FakeLoader:
    """Stands in for datasets.load_dataset -- no network call, no real HF
    download, so this test stays fast and hermetic (CLAUDE.md: tests must
    never depend on external services)."""

    def __init__(self, by_split: dict[str, _FakeDataset]) -> None:
        self._by_split = by_split
        self.calls: list[tuple[str, str, str]] = []

    def __call__(self, name: str, config: str, split: str) -> _FakeDataset:
        self.calls.append((name, config, split))
        return self._by_split[split]


def _make_loader() -> _FakeLoader:
    label_names = ["negative", "neutral", "positive"]
    return _FakeLoader(
        {
            "train": _FakeDataset(
                [{"text": "great job", "label": 2}, {"text": "awful", "label": 0}], label_names
            ),
            "validation": _FakeDataset([{"text": "meh", "label": 1}], label_names),
            "test": _FakeDataset([{"text": "loved it", "label": 2}], label_names),
        }
    )


def test_build_sentiment_splits_renames_validation_to_val() -> None:
    loader = _make_loader()

    df = tweet_eval_data.build_sentiment_splits(loader)

    assert set(df["split"]) == {"train", "val", "test"}
    assert "validation" not in set(df["split"])


def test_build_sentiment_splits_maps_label_ids_to_names() -> None:
    loader = _make_loader()

    df = tweet_eval_data.build_sentiment_splits(loader)

    row = df[df["text"] == "great job"].iloc[0]
    assert row["label"] == "positive"
    assert df[df["text"] == "awful"].iloc[0]["label"] == "negative"


def test_build_sentiment_splits_calls_loader_with_sentiment_config() -> None:
    loader = _make_loader()

    tweet_eval_data.build_sentiment_splits(loader)

    assert all(config == "sentiment" for _, config, _ in loader.calls)
    assert {split for _, _, split in loader.calls} == {"train", "validation", "test"}


def test_build_emotion_splits_uses_emotion_config() -> None:
    loader = _make_loader()

    tweet_eval_data.build_emotion_splits(loader)

    assert all(config == "emotion" for _, config, _ in loader.calls)


def test_ids_are_deterministic_and_unique() -> None:
    loader = _make_loader()

    df = tweet_eval_data.build_sentiment_splits(loader)

    assert df["id"].is_unique
    assert all(row_id.startswith("tweet_eval_sentiment_") for row_id in df["id"])


def test_row_count_matches_total_examples_across_splits() -> None:
    loader = _make_loader()

    df = tweet_eval_data.build_sentiment_splits(loader)

    assert len(df) == 2 + 1 + 1
