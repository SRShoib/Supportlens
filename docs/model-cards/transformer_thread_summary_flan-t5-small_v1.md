# Model card: transformer_thread_summary_flan-t5-small_v1

**Base model:** `google/flan-t5-small`
**Task:** thread (conversation) summarization, free-text generation

## Data & splits

- Datasets: `knkarthick/samsum` + `knkarthick/dialogsum` (HF mirrors of samsum/dialogsum -- the canonical `samsum` repo no longer loads under `datasets>=3.1`, see `docs/decisions.md`), split files `data/splits/{samsum,dialogsum}_v1.parquet`
- Split: each dataset's own fixed train/validation/test partition, used verbatim.
- Training pools both datasets' train rows (`ml/training/train_summarization.py`); ROUGE below is reported per dataset's own test split, not pooled, to stay comparable to each published benchmark (see `docs/decisions.md`).

## Metrics (test split, per dataset)

| Dataset | Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|---|
| samsum_v1 | baseline (`baseline_thread_summary_v1`) | 0.3176 | 0.0893 | 0.2405 |
| samsum_v1 | transformer (`transformer_thread_summary_flan-t5-small_v1`) | 0.4751 | 0.2283 | 0.3922 |
| dialogsum_v1 | baseline (`baseline_thread_summary_v1`) | 0.2566 | 0.0658 | 0.1895 |
| dialogsum_v1 | transformer (`transformer_thread_summary_flan-t5-small_v1`) | 0.4248 | 0.1667 | 0.3425 |

## CPU latency & size

| Dataset | | Transformer | Baseline |
|---|---|---|---|
| samsum_v1 | p50 latency (single request) | 329.4 ms | 0.0 ms |
| dialogsum_v1 | p50 latency (single request) | 326.6 ms | 0.0 ms |
| all | export size | 296.7 MB | no model file (rule-based) |

SPEC §3 CPU summarization latency budget: < 3 s per request (measured/reported, not a hard gate).

## LLM-as-judge (real supportlens tickets)

- n = 50 real ticket summaries judged (`gpt-4o-mini`, 1-5 rubric)
- mean faithfulness: **3.44** / 5
- mean coverage: **2.70** / 5
- 100% of judge responses parsed cleanly

See `docs/summarization-failure-modes.md` for concrete hallucination examples.

## Limitations

- Fine-tuned on samsum/dialogsum -- general and call-center-style dialogue, not customer-support-ticket text specifically. A domain gap analogous to M2's Bitext-synthetic-vs-real-tweets finding is plausible but not separately measured beyond the LLM-judge pass above.
- Free-text generation: no guardrail against hallucinated order numbers, dates, or amounts beyond what the faithfulness rubric measures on a 50-example sample.
- CPU-only inference (SPEC §5: "serving is CPU-only"), no quantization applied.
- Feeds the dashboard's per-ticket summary block (`scripts/compute_thread_summaries.py --model transformer`) -- errors here propagate there.
