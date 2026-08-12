# Model card: transformer_distilbert-base-uncased_v1 (sentiment)

**Base model:** `distilbert-base-uncased`
**Task:** sentiment classification, 3 classes

## Data & splits

- Dataset: `tweet_eval_sentiment_v1` (HF `tweet_eval`, config `sentiment`), split file `data/splits/sentiment_v1.parquet`
- Split: tweet_eval's own fixed train/validation/test partition, used verbatim -- not a fresh seed-42 re-split (see `docs/decisions.md`).
- Test rows evaluated: 12284

## Hyperparameters

| Param | Value |
|---|---|
| num_epochs | 3 |
| batch_size | 32 |
| learning_rate | 2e-05 |
| max_seq_length | 64 |
| warmup_ratio | 0.1 |
| weight_decay | 0.01 |

## Metrics (test split)

- Macro-F1: **0.6835** (baseline `baseline_logistic_regression_v1`: 0.6193)

Per-class F1:

| Class | F1 |
|---|---|
| positive | 0.6678 |
| neutral | 0.6750 |
| negative | 0.7077 |

## CPU latency & size

| | Transformer | Baseline |
|---|---|---|
| p50 latency (single request) | 12.6 ms | 1.7 ms |
| p95 latency | 16.6 ms | 1.9 ms |
| Export size | 256.3 MB | 3.4 MB |

SPEC §3 CPU classification latency budget: < 150 ms per request (measured/reported, not a hard gate).

## Limitations

- Fine-tuned on tweet_eval -- general Twitter text, not customer-support-specific. Applied here to support tickets via transfer learning; a domain gap analogous to M2's Bitext-synthetic-vs-real-tweets finding is plausible but not separately measured here (no human-labeled sentiment/emotion ground truth exists for this repo's own ticket corpus).
- Truncates input to 64 tokens; longer messages lose trailing context.
- CPU-only inference (SPEC §5: "serving is CPU-only"), no quantization applied.
- Feeds `ml/inference/sentiment_trajectory.py`'s per-ticket trajectory and resolution-quality heuristic when `scripts/compute_sentiment_trajectories.py` is run with `--model transformer` -- errors here propagate into that aggregate.
