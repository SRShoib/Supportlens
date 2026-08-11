# Model card: transformer_deberta-v3-small_v1 (intent)

**Base model:** `microsoft/deberta-v3-small`
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
| learning_rate | 1e-05 |
| max_seq_length | 64 |
| warmup_ratio | 0.1 |
| weight_decay | 0.01 |

## Metrics (test split)

- Macro-F1: **0.9970** (baseline `baseline_linear_svc_v1`: 0.9990)

Per-class F1:

| Class | F1 |
|---|---|
| check_invoice | 0.9900 |
| get_invoice | 0.9900 |
| track_order | 0.9932 |
| change_order | 0.9933 |
| registration_problems | 0.9933 |
| change_shipping_address | 0.9966 |
| review | 0.9966 |
| cancel_order | 0.9967 |
| check_payment_methods | 0.9967 |
| complaint | 0.9967 |
| payment_issue | 0.9967 |
| set_up_shipping_address | 0.9967 |
| check_refund_policy | 0.9967 |
| create_account | 0.9967 |
| delivery_period | 0.9967 |
| get_refund | 0.9967 |
| place_order | 0.9967 |
| check_cancellation_fee | 1.0000 |
| contact_customer_service | 1.0000 |
| contact_human_agent | 1.0000 |
| delete_account | 1.0000 |
| delivery_options | 1.0000 |
| edit_account | 1.0000 |
| newsletter_subscription | 1.0000 |
| recover_password | 1.0000 |
| switch_account | 1.0000 |
| track_refund | 1.0000 |

## CPU latency & size

| | Transformer | Baseline |
|---|---|---|
| p50 latency (single request) | 30.0 ms | 1.6 ms |
| p95 latency | 33.0 ms | 1.7 ms |
| Export size | 552.0 MB | 3.4 MB |

SPEC §3 CPU classification latency budget: < 150 ms per request (measured/reported, not a hard gate).

## Limitations

- Fine-tuned on Bitext's synthetic, templated intent instructions — a near-ceiling test score reflects how well the model separates Bitext's 27 templates, not necessarily real customer-message understanding (see `docs/m2-baseline-report.md`'s qualitative spot-check for the same caveat on the classical baseline).
- Truncates input to 64 tokens; longer messages lose trailing context.
- CPU-only inference (SPEC §5: "serving is CPU-only"), no quantization applied.
