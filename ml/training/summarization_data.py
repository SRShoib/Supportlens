"""Builds train/val/test splits for M6's dialogue-summarization transfer
sources: samsum and dialogsum (SPEC M6). Same pattern as
ml/training/tweet_eval_data.py -- each dataset's own native train/validation/
test partition is used verbatim, not re-split with a fresh seed-42 partition
(docs/decisions.md's tweet_eval entry: re-splitting a benchmark's own fixed
split would just be reinventing it, and breaks comparability to published
per-dataset numbers).

The canonical `samsum` HF dataset id no longer loads: it ships as a Python
loading script, and `datasets>=3.1` (this repo's pin) refuses to execute
loading-script datasets at all ("trust_remote_code is not supported
anymore"). `knkarthick/samsum` and `knkarthick/dialogsum` are plain
CSV-backed mirrors with identical {id, dialogue, summary} columns and the
same split sizes as the original benchmarks (samsum: 14731/818/819,
dialogsum: 12460/500/1500) -- see docs/decisions.md.

Writes data/splits/samsum_v1.parquet and data/splits/dialogsum_v1.parquet in
{id, dialogue, summary, split} shape. ml/training/train_summarization.py
pools both train splits; scripts/generate_m6_report.py evaluates ROUGE on
each dataset's own test split separately, per docs/decisions.md.

Run: uv run python -m ml.training.summarization_data
"""

import pandas as pd
from datasets import load_dataset

from ml.training.splits import save_splits

_HF_SPLIT_TO_OURS = {"train": "train", "validation": "val", "test": "test"}


def _build_splits(hf_repo: str, loader: object = load_dataset) -> pd.DataFrame:
    rows = []
    for hf_split, our_split in _HF_SPLIT_TO_OURS.items():
        dataset = loader(hf_repo, split=hf_split)  # type: ignore[operator]
        for example in dataset:
            rows.append(
                {
                    "id": str(example["id"]),
                    "dialogue": example["dialogue"],
                    "summary": example["summary"],
                    "split": our_split,
                }
            )
    return pd.DataFrame(rows)


def build_samsum_splits(loader: object = load_dataset) -> pd.DataFrame:
    return _build_splits("knkarthick/samsum", loader)


def build_dialogsum_splits(loader: object = load_dataset) -> pd.DataFrame:
    return _build_splits("knkarthick/dialogsum", loader)


def main() -> None:
    samsum_df = build_samsum_splits()
    dialogsum_df = build_dialogsum_splits()

    samsum_path = save_splits(samsum_df, "samsum_v1")
    dialogsum_path = save_splits(dialogsum_df, "dialogsum_v1")

    for name, df in [("samsum", samsum_df), ("dialogsum", dialogsum_df)]:
        counts = df["split"].value_counts().to_dict()
        print(f"{name}: {len(df)} rows -> splits={counts}")

    print(f"saved: {samsum_path}, {dialogsum_path}")


if __name__ == "__main__":
    main()
