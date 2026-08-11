# Model card: transformer_distilbert-base-uncased_v1 (intent)

**Base model:** `distilbert-base-uncased`
**Task:** intent classification, 27 classes

## Data & splits

- Dataset: `bitext`, split file `data/splits/intent_v1.parquet`
- Split: stratified 70/15/15 train/val/test, seed 42 (`ml/training/splits.py`) — the identical split M2's classical baseline trained on, so the comparison is apples-to-apples.
- Test rows evaluated: 4031

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

- Macro-F1: **0.9975** (baseline `baseline_linear_svc_v1`: 0.9990)

Per-class F1:

| Class | F1 |
|---|---|
| registration_problems | 0.9868 |
| track_order | 0.9932 |
| get_invoice | 0.9933 |
| payment_issue | 0.9933 |
| check_refund_policy | 0.9933 |
| create_account | 0.9933 |
| check_invoice | 0.9934 |
| review | 0.9966 |
| change_order | 0.9967 |
| delivery_period | 0.9967 |
| get_refund | 0.9967 |
| cancel_order | 1.0000 |
| change_shipping_address | 1.0000 |
| check_cancellation_fee | 1.0000 |
| check_payment_methods | 1.0000 |
| complaint | 1.0000 |
| contact_customer_service | 1.0000 |
| contact_human_agent | 1.0000 |
| delete_account | 1.0000 |
| delivery_options | 1.0000 |
| edit_account | 1.0000 |
| newsletter_subscription | 1.0000 |
| place_order | 1.0000 |
| recover_password | 1.0000 |
| set_up_shipping_address | 1.0000 |
| switch_account | 1.0000 |
| track_refund | 1.0000 |

## CPU latency & size

| | Transformer | Baseline |
|---|---|---|
| p50 latency (single request) | 11.3 ms | 1.6 ms |
| p95 latency | 13.7 ms | 1.7 ms |
| Export size | 256.4 MB | 3.4 MB |

SPEC §3 CPU classification latency budget: < 150 ms per request (measured/reported, not a hard gate).

## Limitations

- Fine-tuned on Bitext's synthetic, templated intent instructions — a near-ceiling test score reflects how well the model separates Bitext's 27 templates, not necessarily real customer-message understanding (see `docs/m2-baseline-report.md`'s qualitative spot-check for the same caveat on the classical baseline).
- Truncates input to 64 tokens; longer messages lose trailing context.
- CPU-only inference (SPEC §5: "serving is CPU-only"), no quantization applied.
