from ml.training import summarization_data


class _FakeDataset:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

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
        self.calls: list[tuple[str, str]] = []

    def __call__(self, repo: str, split: str) -> _FakeDataset:
        self.calls.append((repo, split))
        return self._by_split[split]


def _make_loader() -> _FakeLoader:
    return _FakeLoader(
        {
            "train": _FakeDataset(
                [
                    {"id": "1", "dialogue": "A: hi\nB: hello", "summary": "A greets B."},
                    {"id": "2", "dialogue": "A: bye\nB: bye", "summary": "They say bye."},
                ]
            ),
            "validation": _FakeDataset(
                [{"id": "3", "dialogue": "A: ok\nB: ok", "summary": "They agree."}]
            ),
            "test": _FakeDataset(
                [{"id": "4", "dialogue": "A: wow\nB: yeah", "summary": "They are amazed."}]
            ),
        }
    )


def test_build_samsum_splits_renames_validation_to_val() -> None:
    df = summarization_data.build_samsum_splits(_make_loader())

    assert set(df["split"]) == {"train", "val", "test"}
    assert "validation" not in set(df["split"])


def test_build_samsum_splits_calls_loader_with_knkarthick_mirror() -> None:
    loader = _make_loader()

    summarization_data.build_samsum_splits(loader)

    assert all(repo == "knkarthick/samsum" for repo, _ in loader.calls)
    assert {split for _, split in loader.calls} == {"train", "validation", "test"}


def test_build_dialogsum_splits_calls_loader_with_knkarthick_mirror() -> None:
    loader = _make_loader()

    summarization_data.build_dialogsum_splits(loader)

    assert all(repo == "knkarthick/dialogsum" for repo, _ in loader.calls)


def test_build_samsum_splits_preserves_dialogue_and_summary_text() -> None:
    df = summarization_data.build_samsum_splits(_make_loader())

    row = df[df["id"] == "1"].iloc[0]
    assert row["dialogue"] == "A: hi\nB: hello"
    assert row["summary"] == "A greets B."


def test_ids_are_unique_strings() -> None:
    df = summarization_data.build_samsum_splits(_make_loader())

    assert df["id"].is_unique
    assert all(isinstance(row_id, str) for row_id in df["id"])


def test_row_count_matches_total_examples_across_splits() -> None:
    df = summarization_data.build_samsum_splits(_make_loader())

    assert len(df) == 2 + 1 + 1
