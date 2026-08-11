# Model card: transformer_entities_bert-base-cased_v1

**Base model:** `bert-base-cased`
**Task:** token classification (BIO), 5 entity types: ORDER_ID, PRODUCT, DATE, AMOUNT, ACCOUNT_REF

## Data & splits

- Dataset: `ner_synth_v1` (`ml/data/ner/generate.py`, `data/splits/ner_v1.jsonl`), 70/15/15 train/val/test, seed 42
- Evaluated on: the held-out synthetic test split (offline signal) and the 200-example hand-verified gold set `ner_gold_v1` (the real target metric)
- Gold set rows evaluated: 200
- Synthetic test rows evaluated: 600

## Hyperparameters

| Param | Value |
|---|---|
| num_epochs | 4 |
| batch_size | 16 |
| learning_rate | 3e-05 |
| max_seq_length | 192 |
| warmup_ratio | 0.1 |
| weight_decay | 0.01 |
| include_paraphrases | False |

## Span-level metrics (gold set, exact match on start/end/label)

- Micro-F1: **0.4468** (rules baseline: 0.5850)
- Macro-F1: **0.3890**
- Boundary F1 (label-blind): 0.4468
- Partial F1 (overlap-relaxed): 0.5532

Per-entity F1 (gold set, with 95% bootstrap CI and support):

| Entity | F1 | Precision | Recall | Support | 95% CI |
|---|---|---|---|---|---|
| ORDER_ID | 0.462 | 0.500 | 0.429 | 7 | [0.00, 0.77] |
| PRODUCT | 0.000 | 0.000 | 0.000 | 25 | [0.00, 0.00] |
| DATE | 0.497 | 0.672 | 0.394 | 109 | [0.40, 0.59] |
| AMOUNT | 0.565 | 0.765 | 0.448 | 29 | [0.33, 0.74] |
| ACCOUNT_REF | 0.421 | 0.333 | 0.571 | 7 | [0.00, 0.73] |

## Domain gap: synthetic test vs. gold set

- Synthetic test micro-F1: 0.9911
- Gold set micro-F1: 0.4468
- Gap: +0.5443

## CPU latency & size

| | This model | Rules baseline |
|---|---|---|
| p50 latency (single request) | 37.2 ms | 0.1 ms |
| p95 latency | 40.8 ms | 0.1 ms |
| Export size | 411.8 MB | n/a (no file) |

SPEC §3 CPU NER latency budget: < 250 ms per request (measured/reported, not a hard gate).

## Limitations

- Trained on synthetic data (templates + real-shell injection, `ml/data/ner/generate.py`), evaluated for real on a single-annotator, 200-example gold set -- small enough that per-entity F1 deltas under ~0.10 are noise, not signal (see the bootstrap CIs above).
- Truncates input to 192 tokens.
- Spans are only valid against `clean_text`-shaped input -- offsets are relative to exactly the string passed to `predict()`, never a re-cleaned copy of it.
- Subword labelling: every subword overlapping a gold span is labelled (B- on the first, I- on the rest), matching the decode path exactly rather than a word-level scheme.
- CPU-only inference (SPEC §5: "serving is CPU-only"), no quantization applied.
