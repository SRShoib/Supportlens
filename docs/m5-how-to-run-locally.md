# M5: running the sentiment/emotion fine-tunes locally

Same shape as `docs/m3-how-to-run-locally.md` and `docs/m4-how-to-run-locally.md`: CLAUDE.md ground rule #3
means GPU training never happens in Docker, CI, or anything Claude Code executes directly. Everything below
has been smoke-tested on CPU (`--max-steps 5`, no real learning, `CUDA_VISIBLE_DEVICES=""` forced explicitly
even though this machine's torch build is CUDA-enabled — the smoke test is a code-path check, not a training
run, so it deliberately stays off the GPU) to confirm training ran, per-epoch macro-F1 eval ran, and the
export landed at the path `apps/api/routers/predict.py`'s `_TRANSFORMER_MODEL_DIRS` expects. The numbers from
a real run on the full tweet_eval splits are the ones that matter.

## 0. Prerequisite: the tweet_eval splits must exist

Unlike M3 (which trained on M1's ingested corpus), M5 trains on tweet_eval directly — a transfer source
(SPEC §2), never ingested into Postgres:

```bash
make sentiment-emotion-data    # writes data/splits/sentiment_v1.parquet and emotion_v1.parquet
```

This downloads tweet_eval's own fixed train/validation/test splits from the HF Hub (no re-splitting — see
`docs/decisions.md`) and writes them in the same `{id, text, label, split}` shape M2's `data/splits/*.parquet`
files use. Already run once in this session: sentiment has 59,899 rows (45,615 train / 2,000 val / 12,284
test, 3 classes); emotion has 5,052 rows (3,257 train / 374 val / 1,421 test, 4 classes).

## 1. One-time setup

If you already did this for M3/M4, skip to step 2 — it's the same `training` group.

```bash
make install-training          # uv sync --group training: transformers, accelerate,
                                # evaluate, sentencepiece, tiktoken, protobuf, pyyaml
```

Read `docs/decisions.md`'s 2026-08-11 entries first if your torch is already the CUDA build — `uv sync
--group training` has twice silently swapped the CUDA wheel for the CPU-only one on this machine. After
running it, always verify:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If `torch.cuda.is_available()` is `False`, restore the CUDA build (RTX 4060 Ti, driver 591.86, per
`docs/decisions.md`):

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
```

## 2. Classical baselines first (already run for real, not just smoke-tested)

```bash
make train-baseline-sentiment  # test macro_f1 = 0.6193 (logistic_regression)
make train-baseline-emotion    # test macro_f1 = 0.6541 (linear_svc)
```

Both already ran against the real tweet_eval test splits and exported to `models/baseline_sentiment_v1/` and
`models/baseline_emotion_v1/` — these are real numbers, not placeholders, persisted as `EvalRun` rows
(CLAUDE.md rule #5).

## 3. The two transformer runs

One parameterized script, two configs (task fixed at classification, model varies), same script M3 already
uses unchanged:

```bash
uv run python ml/training/train_transformer.py --config ml/training/configs/sentiment_distilbert.yaml
uv run python ml/training/train_transformer.py --config ml/training/configs/emotion_distilbert.yaml
```

Each trains on `data/splits/{task}_v1.parquet` from step 0, evaluates macro-F1 on val each epoch, evaluates
once more on the held-out test split at the end, and exports to
`models/transformer_{task}_distilbert-base-uncased_v1/final/`.

**A CPU smoke-test export already sits at both paths** (from the `--max-steps 5` runs above) — the real GPU
run will overwrite it, same convention M3/M4 followed. Until you run the real training, `model="transformer"`
on `/predict/sentiment` / `/predict/emotion` will serve untrained (near-random) predictions, not an error —
there's nothing to catch that case programmatically the way a missing file 503s.

**Why distilbert-base-uncased for both, unlike M4's cased NER models:** casing isn't a strong signal for
sentiment/emotion the way it is for entity recognition, so this follows M3's intent/urgency precedent rather
than M4's.

## 4. Expected VRAM and time (RTX 4060 Ti, 16GB)

Rough expectations, scaled from M3's real DistilBERT numbers by train-rows x epochs — **not measured on this
hardware** (this machine ran only the CPU smoke tests):

| Run | Params | Train rows | Epochs | Expected VRAM | Expected time |
|---|---|---|---|---|---|
| sentiment + DistilBERT | 66M | 45,615 | 3 | ~2-3 GB | ~8-12 min |
| emotion + DistilBERT | 66M | 3,257 | 4 | ~2-3 GB | ~1-3 min |

Sanity-check a config on your GPU before committing to a full run (also proves CUDA is actually being used,
unlike the CPU smoke test above):

```bash
uv run python ml/training/train_transformer.py --config ml/training/configs/sentiment_distilbert.yaml --max-steps 5
```

## 5. After a run finishes

1. Check the printed `test macro_f1` against the baselines in step 2 (0.6193 sentiment, 0.6541 emotion).
2. Tell Claude Code the run is done — `scripts/generate_m5_report.py` (not yet run for real; it needs these
   exports to exist) will persist the comparison `EvalRun` rows, render `docs/m5-comparison-report.md`, and
   write model cards.
3. Re-run the trajectory backfill with the trained transformer instead of the baseline:
   ```bash
   uv run python scripts/compute_sentiment_trajectories.py --model transformer
   ```
   (defaults to `--model baseline` otherwise — see that script's module docstring for why).
4. If a run's test macro-F1 comes back *worse* than the baseline, that's a valid, reportable outcome
   (CLAUDE.md rule #2: the comparison is the deliverable) — don't discard the run.
