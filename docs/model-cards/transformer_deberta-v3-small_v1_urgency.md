# Model card: transformer_deberta-v3-small_v1 (urgency)

**Base model:** `microsoft/deberta-v3-small`
**Task:** urgency classification, 3 classes

## Data & splits

- Dataset: `twitter_slice_v1`, split file `data/splits/urgency_v1.parquet`
- Split: stratified 70/15/15 train/val/test, seed 42 (`ml/training/splits.py`) — the identical split M2's classical baseline trained on, so the comparison is apples-to-apples.
- Test rows evaluated: 12678

## Hyperparameters

| Param | Value |
|---|---|
| num_epochs | 2 |
| batch_size | 32 |
| learning_rate | 1e-05 |
| max_seq_length | 128 |
| warmup_ratio | 0.1 |
| weight_decay | 0.01 |

## Metrics (test split)

- Macro-F1: **0.9095** (baseline `baseline_linear_svc_v1`: 0.7944)

Per-class F1:

| Class | F1 |
|---|---|
| medium | 0.8716 |
| high | 0.8831 |
| low | 0.9738 |

## CPU latency & size

| | Transformer | Baseline |
|---|---|---|
| p50 latency (single request) | 29.4 ms | 1.6 ms |
| p95 latency | 31.8 ms | 1.7 ms |
| Export size | 551.9 MB | 3.5 MB |

SPEC §3 CPU classification latency budget: < 150 ms per request (measured/reported, not a hard gate).

## Limitations

- Trained on `weak_label_urgency()` rule-based labels, not human-verified ground truth — test macro-F1 measures agreement with the weak-label heuristic, not objective urgency. See the Cohen's kappa section of `docs/m2-baseline-report.md` for how the weak labels compare to a small LLM-labeled sample.
- Truncates input to 128 tokens; longer messages lose trailing context.
- CPU-only inference (SPEC §5: "serving is CPU-only"), no quantization applied.
