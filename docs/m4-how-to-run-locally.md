# M4: running the NER fine-tunes locally

Same shape as `docs/m3-how-to-run-locally.md`: CLAUDE.md ground rule #3 means GPU training never happens in
Docker, CI, or anything Claude Code executes directly. Everything below has been smoke-tested on CPU
(`--max-steps 5`, a tiny 29-example template-only dataset, no real learning) to confirm the code path itself
works — training ran, per-epoch span-F1 eval ran, the export landed, and `TokenClassificationPredictor`
loaded that export and produced offset-correct (if meaningless, from 5 untrained steps) predictions. The
numbers from a real run on the full dataset are the ones that matter.

## 0. Prerequisite: the synthetic dataset must exist

Unlike M3 (which trained on M1's already-ingested corpus), M4 first needs its own dataset generated:

```bash
make ner-data          # needs an ingested Twitter slice (M1) already in Postgres
```

This writes `data/splits/ner_v1.jsonl` (~4,000 examples) and `data/splits/ner_pools_v1.json` (the seeded
shell/gold partition scripts/ner_gold_export.py reads later). Both are gitignored — regenerate any time,
deterministically, from the same seed.

## 1. One-time setup

If you already did this for M3, skip to step 2 — it's the same `training` group.

```bash
# From the repo root, with your normal venv already active:
make install-training          # uv sync --group training: transformers, accelerate,
                                # evaluate, sentencepiece, tiktoken, protobuf, pyyaml
```

**Read this before running the command above if your torch is already the CUDA build.**
`docs/decisions.md`'s 2026-08-11 entries document a real trap, confirmed twice now: `accelerate` (pulled in
by `training`) requires `torch` with no version constraint, and this project's `[tool.uv.sources]` pins
`torch` to the CPU-only wheel **project-wide, not group-scoped** — so `uv sync --group training` can just as
easily strip a manually-installed CUDA build as `--group serving` can. After running it, always check:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If `torch.cuda.is_available()` is `False`, restore the CUDA build:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Your GPU/driver (per `docs/decisions.md`): **RTX 4060 Ti, 16GB VRAM, driver 591.86** — cu124 is the
broadly-compatible choice for any reasonably current driver.

## 2. The two runs

One parameterized script, two configs (task is fixed at `entities`, model varies):

```bash
uv run python ml/training/train_token_classification.py --config ml/training/configs/ner/distilbert_cased.yaml
uv run python ml/training/train_token_classification.py --config ml/training/configs/ner/bert_cased.yaml
```

(`make train-ner` runs the first of these — `distilbert_cased.yaml` — as a shortcut; run the second directly.)

Each one loads `data/splits/ner_v1.jsonl`, tokenizes with the model's own (cased) tokenizer using
`return_offsets_mapping=True`, projects the gold char spans onto per-token BIO labels
(`ml/training/train_token_classification.py::align_labels`), trains with dynamic padding
(`DataCollatorForTokenClassification`), evaluates span-level micro-F1 each epoch via the exact same decoder
`ml/inference/token_classification.py::decode_spans` that serves predictions in production, evaluates once
more on the held-out test split at the end, and exports to
`models/transformer_entities_{model}_v1/final/` — safetensors weights, tokenizer, config, plus a
`label_map.json` recording the BIO scheme and label order (read by `TokenClassificationPredictor` so label
order is an explicit, independently-checkable artifact rather than an HF-internal detail).

**Why cased models, unlike M3's `-uncased`:** casing is a strong NER signal (`ORD-99321`, `iPhone 14 Pro`,
`Black Friday`) that lowercasing would throw away — a per-task decision, not an inconsistency with M3.
**Why not DeBERTa-v3-small** (M3's urgency winner): its SentencePiece offset mapping is fiddlier, and offset
correctness is load-bearing here in a way it never was for classification — plus this repo already hit and
documented a real transformers-5.x tokenizer-conversion crash specific to that model. Not worth the risk for
a second variant when `bert-base-cased` gives a clean comparison point with zero additional tokenizer risk.

## 3. Expected VRAM and time (RTX 4060 Ti, 16GB)

Rough expectations, **not measured on this hardware** (this machine ran only the CPU smoke test) — scale
from M3's numbers, adjusted for ~4,000 examples (vs. M3's 18k-59k) and longer sequences (192 vs. 64-128):

| Run | Params | Train rows (~70% of 4,000) | Epochs | Expected VRAM | Expected time |
|---|---|---|---|---|---|
| entities + DistilBERT (cased) | 66M | ~2,800 | 4 | ~2-3 GB | ~2-4 min |
| entities + BERT (cased) | 108M | ~2,800 | 4 | ~3-4 GB | ~4-6 min |

If `include_paraphrases: true` is turned on in a config (default `false`), train rows roughly double —
scale time accordingly.

To sanity-check a config on your GPU before committing to a full run:

```bash
uv run python ml/training/train_token_classification.py --config ml/training/configs/ner/distilbert_cased.yaml --max-steps 5
```

## 4. Two traps this repo already knows about, that apply here too

- **`TrainingArguments` doesn't take `warmup_ratio`** in the pinned transformers version (only
  `warmup_steps`) — `train_token_classification.py` computes `warmup_steps` from the config's `warmup_ratio`
  manually, same as `train_transformer.py`. Nothing to do; noted here so a `TypeError` about `warmup_ratio`
  after a dependency bump isn't a surprise.
- **transformers 5.x has a tokenizer-conversion bug specific to `microsoft/deberta-v3-small`** — irrelevant
  to the two configs above (neither uses DeBERTa, precisely to avoid this), but if a third variant is ever
  added, re-read `docs/decisions.md`'s 2026-08-10 entry first.

## 5. After a run finishes

1. Check the printed `test span_micro_f1` — a first sanity signal only; the real per-entity numbers (and the
   rules-vs-model comparison) come from `scripts/generate_m4_report.py` once it exists, run against both this
   export and the hand-verified gold set.
2. Tell Claude Code the run is done. The follow-up work (model card, the rules-vs-model comparison report)
   reads from the exported `models/transformer_entities_*_v1/final/` artifacts and needs them to exist first.
3. If a run's test span-F1 comes back *worse* than the rules baseline on some entity type, that's a valid,
   reportable outcome per SPEC M4's accept criterion ("compare against a regex/rules baseline... which will
   genuinely win on ORDER_ID — say so") — don't discard the run.
4. Separately, hand-annotate the 200-example gold set (`make ner-gold-export` → edit
   `data/gold/ner_gold_v1.todo.md` per `docs/ner-annotation-guidelines.md` → `make ner-gold-import`) if that
   hasn't happened yet — the comparison report needs both the trained model and the gold set.
