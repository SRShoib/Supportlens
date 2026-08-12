# M6: running the FLAN-T5-small thread-summarization fine-tune locally

Same shape as `docs/m3-how-to-run-locally.md` / `docs/m4-how-to-run-locally.md` / `docs/m5-how-to-run-locally.md`:
CLAUDE.md ground rule #3 means GPU training never happens in Docker, CI, or anything Claude Code executes
directly. Everything below has been smoke-tested on CPU (`--max-steps 5`, no real learning,
`CUDA_VISIBLE_DEVICES=""` forced explicitly even though this machine's torch build is CUDA-enabled — the
smoke test is a code-path check, not a training run, so it deliberately stays off the GPU) to confirm
training ran, per-epoch ROUGE eval ran, and the export landed where `apps/api/routers/predict.py`'s
`_SUMMARY_TRANSFORMER_MODEL_DIR` and `scripts/compute_thread_summaries.py`'s `SUMMARY_TRANSFORMER_DIR` both
expect. The numbers from a real run on the full samsum + dialogsum train splits are the ones that matter.

## 0. Prerequisite: the samsum/dialogsum splits must exist

```bash
make summarization-data    # writes data/splits/samsum_v1.parquet and dialogsum_v1.parquet
```

**The canonical `samsum` HF dataset id no longer loads** under this repo's `datasets>=3.1` pin — it ships as
a Python loading script, and recent `datasets` versions refuse to execute those at all
(`trust_remote_code is not supported anymore`). `ml/training/summarization_data.py` uses `knkarthick/samsum`
and `knkarthick/dialogsum` instead — plain CSV-backed mirrors with identical `{id, dialogue, summary}`
columns and the same split sizes as the original benchmarks. See `docs/decisions.md`.

Already run once in this session, real numbers: samsum has 16,368 rows (14,731 train / 818 val / 819 test);
dialogsum has 14,460 rows (12,460 train / 500 val / 1,500 test).

## 1. One-time setup

If you already did this for M3/M4/M5, skip to step 2 — it's the same `training` group, plus one new
lightweight dependency (`rouge-score`, added to the default `ml` group, not `training` — it's a real
CI-tested dependency of `ml/evaluation/rouge_metrics.py`, so it's already there after a plain `uv sync`).

```bash
make install-training          # uv sync --group training
```

**Read `docs/decisions.md`'s 2026-08-11 and 2026-08-12 entries before running any `uv sync` variant on this
machine.** Two traps, both hit for real while building M6:

1. `uv sync --group training` (or `--group serving`) silently swaps the CUDA torch build for a CPU-only one.
2. A **bare `uv sync`** (no `--group` flag at all) is worse — it *uninstalls* `torch`/`transformers`/
   `accelerate` outright, because they're only present via a non-default group. This happened while adding
   `rouge-score` to the default `ml` group in this same session.

After *any* `uv sync`, verify:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If `torch.cuda.is_available()` is `False`, restore the CUDA build (RTX 4060 Ti, driver 591.86, per
`docs/decisions.md`):

```bash
uv pip install --reinstall torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

**Also stop any running `make dev` / uvicorn process before syncing.** A live dev server holding
`transformers`/`safetensors` open can leave `uv sync` unable to replace a locked compiled extension,
resulting in a half-upgraded `safetensors` that fails on import later with an unrelated-looking
`ImportError: cannot import name 'TensorSpec'`. Fix (if it happens anyway):
`uv pip install --reinstall safetensors`.

## 2. Classical baseline (no training needed — it's rule-based)

Lead-k extractive summarization (`ml/inference/extractive_summary.py::ExtractiveSummaryPredictor`) has no
model file to train or export. `DEFAULT_K=4`, already picked for real against the pooled samsum+dialogsum
val split (see `docs/decisions.md`) — nothing to run here before the transformer step.

## 3. The FLAN-T5-small fine-tune

```bash
uv run python ml/training/train_summarization.py \
    --config ml/training/configs/thread_summary_flan_t5_small.yaml
```

Pools samsum + dialogsum's `train` rows (27,191 rows total), evaluates ROUGE-1/2/L on the pooled `val` split
each epoch, and exports to `models/transformer_thread_summary_flan-t5-small_v1/final/`. **Never touches
either dataset's `test` split** — that's deliberately left to `scripts/generate_m6_report.py`, evaluated
per-dataset (not pooled), so the numbers stay comparable to each published benchmark (see
`docs/decisions.md`).

**A CPU smoke-test export already sits at that path** (from a `--max-steps 5` run) — the real GPU run will
overwrite it, same convention M3/M4/M5 followed. The smoke-test run printed `There were missing keys in the
checkpoint model loaded: ['encoder.embed_tokens.weight', 'decoder.embed_tokens.weight']` — this is benign:
T5 ties its input/output embeddings, `save_only_model=True` doesn't re-save the tied copies, and `.from_pretrained`
re-ties them automatically on load (confirmed: the smoke-test export loads and generates a coherent,
on-topic summary via `SummarizationPredictor`, not garbage). Don't be alarmed if you see it again on the
real run.

Until you run the real training, `model="transformer"` on `/predict/summary` (and
`scripts/compute_thread_summaries.py --model transformer`) will serve the smoke-test model's near-random
output, not an error — there's nothing to catch that case programmatically the way a missing file 503s.

## 4. Expected VRAM and time (RTX 4060 Ti, 16GB)

Rough expectations, scaled from M3/M5's real DistilBERT numbers by param count and train-rows x epochs —
**not measured on this hardware** (this machine ran only the CPU smoke test):

| Run | Params | Train rows (pooled) | Epochs | Expected VRAM | Expected time |
|---|---|---|---|---|---|
| thread_summary + FLAN-T5-small | 77M | 27,191 | 3 | ~3-4 GB | ~20-35 min |

Seq2seq generation during eval (beam search, `num_beams=4`) is slower per-example than classification's
argmax, so epoch eval will take noticeably longer relative to train time than M3/M5's classifiers did.

Sanity-check the config on your GPU before committing to a full run:

```bash
uv run python ml/training/train_summarization.py \
    --config ml/training/configs/thread_summary_flan_t5_small.yaml --max-steps 5
```

## 5. After a run finishes

1. Check the printed `final val rouge1/rouge2/rougeL` — there's no baseline number to compare against yet
   at this step (the baseline is evaluated for the first time in step 3 below, on the real *test* splits,
   not val).
2. Re-run the thread-summary backfill with the trained transformer instead of the baseline, against this
   repo's real ingested tickets:
   ```bash
   make predict-summary          # or: uv run python scripts/compute_thread_summaries.py --model transformer
   ```
   (defaults to `--model baseline` otherwise — see that script's module docstring for why, same reasoning
   M5's trajectory backfill uses).
3. Run the LLM-as-judge pass on 50 of the real summaries the step above just wrote (SPEC M6, budget ≈ $0.30
   — this session's actual cumulative spend so far is ~$0.03 out of the $5 project cap, so there's plenty
   of headroom):
   ```bash
   # cheap dry run first (a few cents at most):
   uv run python -m ml.data.llm_judge_summaries --limit 5
   # then the full 50:
   make judge-summaries
   ```
   Requires `OPENAI_API_KEY` and `LLM_ENABLED=true` in `.env` — refuses to spend anything otherwise.
4. Tell Claude Code the run is done — `scripts/generate_m6_report.py` (**deliberately not yet run for real**
   in this session, only import/syntax-checked — it needs the real exports + judge Predictions from steps
   2-3 to produce real numbers, mirroring exactly how `scripts/generate_m5_report.py` was only run once,
   after M5's real GPU training) will persist the comparison `EvalRun` rows (per-dataset ROUGE + the judge
   aggregate), render `docs/m6-comparison-report.md`, write the model card, and print the lowest-faithfulness
   real-ticket examples to stdout:
   ```bash
   make eval-summarization
   ```
5. Read those printed low-faithfulness examples and write `docs/summarization-failure-modes.md` with ≥3 real
   failure examples (SPEC M6's accept criterion) — this step needs a human (or a later Claude Code turn) to
   actually read the hallucinated output and describe it; the report script only surfaces candidates, it
   doesn't write that doc.
6. If a run's test ROUGE comes back *worse* than the lead-k baseline, that's a valid, reportable outcome
   (CLAUDE.md rule #2: the comparison is the deliverable) — don't discard the run.
