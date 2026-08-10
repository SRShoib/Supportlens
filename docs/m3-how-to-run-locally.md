# M3: running the transformer fine-tunes locally

This is the part of M3 that runs on your machine, on your NVIDIA GPU — CLAUDE.md is explicit that GPU
training never happens in Docker, CI, or anything Claude Code executes directly. Everything below has been
smoke-tested on CPU (`--max-steps N`, a handful of steps, no real learning) to confirm the code path itself
works; the numbers you'll get from a real run are the ones that matter.

## 1. One-time setup

```bash
# From the repo root, with your normal venv already active (uv sync already run for M0-M2):
make install-training          # uv sync --group training: transformers, accelerate,
                                # evaluate, sentencepiece, tiktoken, protobuf, pyyaml
```

`make install-training` pulls in a **CPU-only** torch build too, transitively (via `accelerate`, which
hard-requires torch). Replace it with the CUDA build matching your driver:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Your GPU/driver at the time this was written: **RTX 4060 Ti, 16GB VRAM, driver 591.86** — cu124 is the
broadly-compatible choice for any reasonably current driver. Confirm afterward:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If `torch.cuda.is_available()` is `False`, the CUDA wheel didn't take — re-run the install command above
(sometimes uv's cache serves the CPU wheel back; add `--reinstall` if so).

## 2. The four runs

One parameterized script, four configs (task x model):

```bash
uv run python ml/training/train_transformer.py --config ml/training/configs/intent_distilbert.yaml
uv run python ml/training/train_transformer.py --config ml/training/configs/intent_deberta.yaml
uv run python ml/training/train_transformer.py --config ml/training/configs/urgency_distilbert.yaml
uv run python ml/training/train_transformer.py --config ml/training/configs/urgency_deberta.yaml
```

Each one trains on the exact same `data/splits/{task}_v1.parquet` M2's baselines used (regenerate with
`make build-splits` first if you haven't run M2 yet, or if the underlying data has changed), evaluates on
val each epoch, evaluates once more on the held-out test split at the end, and exports to
`models/transformer_{task}_{model}_v1/final/` (HF `save_pretrained` format — safetensors weights, tokenizer,
config, plus a `label_map.json` this script writes so the inference wrapper doesn't need to re-derive label
order later).

## 3. Expected VRAM and time (RTX 4060 Ti, 16GB)

Both models are small enough that VRAM is not the constraint — batch sizes in the configs (32) were chosen
for throughput, not to fit under a ceiling. Rough expectations, **not measured on this hardware** (this
machine ran only CPU smoke tests):

| Run | Params | Train rows | Epochs | Expected VRAM | Expected time |
|---|---|---|---|---|---|
| intent + DistilBERT | 66M | 18,810 | 3 | ~2-3 GB | ~3-5 min |
| intent + DeBERTa-v3-small | ~140M | 18,810 | 3 | ~3-4 GB | ~6-10 min |
| urgency + DistilBERT | 66M | 59,163 | 2 | ~2-3 GB | ~8-12 min |
| urgency + DeBERTa-v3-small | ~140M | 59,163 | 2 | ~3-4 GB | ~15-20 min |

If you want to sanity-check a config before committing to a full run (the same thing done here on CPU, but
on your GPU so it also proves CUDA is actually being used):

```bash
uv run python ml/training/train_transformer.py --config ml/training/configs/intent_distilbert.yaml --max-steps 5
```

## 4. Two real bugs the CPU smoke tests caught before this ever reached you

Documented here so you don't waste time rediscovering them if a dependency update reintroduces either:

- **`TrainingArguments` doesn't take `warmup_ratio`** in the transformers version this project currently
  pins (only `warmup_steps`) — the script computes `warmup_steps` from the config's `warmup_ratio` manually,
  so this is already handled; you don't need to do anything, but if you see a `TypeError` about
  `warmup_ratio` after a dependency bump, this is why.
- **transformers 5.x (as of when this was written) has a tokenizer-conversion bug specific to
  `microsoft/deberta-v3-small`** — it misroutes the SentencePiece model file through a tiktoken BPE parser
  and crashes. `pyproject.toml`'s `training` group pins `transformers<5.0` specifically because of this. If
  you deliberately upgrade past 5.0 later, re-run the DeBERTa smoke test above first.

## 5. After a run finishes

1. Check the printed `test macro_f1` against M2's baseline numbers in `docs/m2-baseline-report.md`.
2. Tell Claude Code the run is done — the follow-up work (model card in `docs/model-cards/`, the
   baseline-vs-transformer comparison report, wiring the API's baseline/transformer flag, ONNX quantization
   if latency needs it) all read from the exported `models/transformer_*_v1/final/` artifacts and needs
   them to exist first.
3. If a run's test macro-F1 comes back *worse* than the baseline, that's a valid, reportable outcome per
   SPEC M3's accept criteria ("or the report honestly says it doesn't and why") — don't discard the run.
