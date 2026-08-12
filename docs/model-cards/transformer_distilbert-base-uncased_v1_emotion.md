# Model card: transformer_distilbert-base-uncased_v1 (emotion)

**Base model:** `distilbert-base-uncased`
**Task:** emotion classification, 4 classes

## Data & splits

- Dataset: `tweet_eval_emotion_v1` (HF `tweet_eval`, config `emotion`), split file `data/splits/emotion_v1.parquet`
- Split: tweet_eval's own fixed train/validation/test partition, used verbatim -- not a fresh seed-42 re-split (see `docs/decisions.md`).
- Test rows evaluated: 1421

## Hyperparameters

| Param | Value |
|---|---|
| num_epochs | 4 |
| batch_size | 32 |
| learning_rate | 2e-05 |
| max_seq_length | 64 |
| warmup_ratio | 0.1 |
| weight_decay | 0.01 |

## Metrics (test split)

- Macro-F1: **0.7598** (baseline `baseline_linear_svc_v1`: 0.6541)

Per-class F1:

| Class | F1 |
|---|---|
| optimism | 0.5664 |
| sadness | 0.7974 |
| joy | 0.8206 |
| anger | 0.8549 |

## CPU latency & size

| | Transformer | Baseline |
|---|---|---|
| p50 latency (single request) | 11.6 ms | 1.7 ms |
| p95 latency | 13.0 ms | 2.1 ms |
| Export size | 256.3 MB | 2.6 MB |

SPEC §3 CPU classification latency budget: < 150 ms per request (measured/reported, not a hard gate).

## Limitations

- Fine-tuned on tweet_eval -- general Twitter text, not customer-support-specific. Applied here to support tickets via transfer learning; a domain gap analogous to M2's Bitext-synthetic-vs-real-tweets finding is plausible but not separately measured here (no human-labeled sentiment/emotion ground truth exists for this repo's own ticket corpus).
- Truncates input to 64 tokens; longer messages lose trailing context.
- CPU-only inference (SPEC §5: "serving is CPU-only"), no quantization applied.
