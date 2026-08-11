"""Builds train/val/test splits for M5's transfer-learning sources: tweet_eval's
"sentiment" (3-class) and "emotion" (4-class) configs (SPEC M5). Unlike
ml/training/splits.py, these are never re-split with a fresh stratified
70/15/15 partition -- tweet_eval already ships fixed, canonical train/
validation/test splits, and re-splitting them would just be reinventing the
benchmark rather than reproducing it. HF's "validation" is renamed to "val"
to match this repo's own {train,val,test} convention; seed=42 has nothing to
do here since no random partitioning happens (see docs/decisions.md).

Writes data/splits/sentiment_v1.parquet and data/splits/emotion_v1.parquet in
the same {id, text, label, split} shape ml/training/splits.py's parquet files
use, so ml/training/train_transformer.py and the classical baseline scripts
(ml/training/train_baseline_*.py) work against them completely unchanged.

Run: uv run python -m ml.training.tweet_eval_data
"""

import pandas as pd
from datasets import load_dataset

from ml.training.splits import save_splits

_HF_SPLIT_TO_OURS = {"train": "train", "validation": "val", "test": "test"}


def _build_splits(config: str, loader: object = load_dataset) -> pd.DataFrame:
    rows = []
    for hf_split, our_split in _HF_SPLIT_TO_OURS.items():
        dataset = loader("tweet_eval", config, split=hf_split)  # type: ignore[operator]
        label_names = dataset.features["label"].names
        for i, example in enumerate(dataset):
            rows.append(
                {
                    "id": f"tweet_eval_{config}_{hf_split}_{i}",
                    "text": example["text"],
                    "label": label_names[example["label"]],
                    "split": our_split,
                }
            )
    return pd.DataFrame(rows)


def build_sentiment_splits(loader: object = load_dataset) -> pd.DataFrame:
    """3-class: negative/neutral/positive (SPEC M5)."""
    return _build_splits("sentiment", loader)


def build_emotion_splits(loader: object = load_dataset) -> pd.DataFrame:
    """4-class: anger/joy/optimism/sadness (SPEC M5)."""
    return _build_splits("emotion", loader)


def main() -> None:
    sentiment_df = build_sentiment_splits()
    emotion_df = build_emotion_splits()

    sentiment_path = save_splits(sentiment_df, "sentiment_v1")
    emotion_path = save_splits(emotion_df, "emotion_v1")

    for name, df in [("sentiment", sentiment_df), ("emotion", emotion_df)]:
        counts = df["split"].value_counts().to_dict()
        label_counts = df["label"].value_counts().to_dict()
        print(f"{name}: {len(df)} rows -> splits={counts} labels={label_counts}")

    print(f"saved: {sentiment_path}, {emotion_path}")


if __name__ == "__main__":
    main()
