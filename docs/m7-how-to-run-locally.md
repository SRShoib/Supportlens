# M7: running the topic-discovery pipeline locally

Same shape as `docs/m3-how-to-run-locally.md` / `docs/m5-how-to-run-locally.md` /
`docs/m6-how-to-run-locally.md`: CLAUDE.md ground rule #3 means this never runs in Docker, CI, or anything
Claude Code executes directly. Everything below has been code-path-checked (unit tests over
`ml/evaluation/trend_metrics.py` and `ml/evaluation/topic_metrics.py`, an integration test over
`scripts/assign_topics.py`'s DB writes, and `scripts/generate_m7_report.py` run end-to-end against small
hand-built fake artifacts to confirm the report/model-card renderers and `EvalRun` persistence work) — but
**no BERTopic/UMAP/HDBSCAN fit has actually happened yet**. This machine has no `topics` group installed and
this session had no GPU access; the numbers from a real run on the full Twitter slice are the ones that
matter, and `docs/m7-comparison-report.md` / `docs/model-cards/topics_bertopic_v1.md` don't exist until you
produce them in step 5.

Unlike M3/M5/M6, this module is **unsupervised** — there's no accuracy/ROUGE number to fine-tune toward, and
UMAP/HDBSCAN fitting is CPU work (only the embedding step benefits from GPU). Expect this to be more of an
iterate-on-hyperparameters loop than a single training run.

## 1. One-time setup

```bash
make install-topics          # uv sync --group topics (sentence-transformers, bertopic, umap-learn, hdbscan)
```

**Read `docs/decisions.md`'s CUDA-torch-swap entries before running any `uv sync` variant on this machine** —
the exact same trap M3/M5/M6 documented applies here: `sentence-transformers` pulls in torch as a hard
dependency, and a plain `uv sync --group topics` resolves it to the CPU wheel via this repo's `pytorch-cpu`
source, same as `serving`. After syncing, verify:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If `torch.cuda.is_available()` is `False` and you want GPU-accelerated embedding, restore the CUDA build
(RTX 4060 Ti, driver 591.86, per `docs/decisions.md`):

```bash
uv pip install --reinstall torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

**Also stop any running `make dev` / uvicorn process before syncing** — same locked-compiled-extension trap
already documented for `transformers`/`safetensors`.

## 2. Embed the corpus

```bash
make embed-tickets          # uv run python scripts/compute_embeddings.py
```

Embeds every Twitter ticket's **customer messages only** (see `scripts/compute_embeddings.py`'s docstring
for why) with `sentence-transformers/all-MiniLM-L6-v2`, writing
`data/embeddings/tickets_minilm_v1.{npy,parquet}`. ~36,579 tickets (SPEC's real-ticket corpus, Bitext
excluded — see `docs/decisions.md`), ~2 minutes on CPU, faster with GPU torch installed. Smoke-test first with
`--limit 500` if you just want to confirm the code path before committing to the full embed.

## 3. Fit both topic-model variants

```bash
make fit-topics             # uv run python ml/training/topic_model.py --config ml/training/configs/topics_minilm_bertopic.yaml
```

Fits the TF-IDF/KMeans baseline (CLAUDE.md rule #2 — see `docs/decisions.md`) and BERTopic
(UMAP + HDBSCAN) from the same embeddings, both labeled via c-TF-IDF over the same documents, and exports
`models/topics_{kmeans,bertopic}_v1/{topics.json,assignments.parquet}`.

**This is the step most likely to need hyperparameter iteration.** SPEC M7's acceptance bar is ≥ 30 topics
(excluding the HDBSCAN outlier cluster). `ml/training/configs/topics_minilm_bertopic.yaml`'s
`hdbscan_min_cluster_size: 75` is a starting guess against ~36k documents, not a tuned value — if the real
run comes back with too few topics (a handful of huge clusters) or too many (mostly noise), lower or raise
it and re-run this step. `kmeans_n_clusters: 40` is a comparable fixed-k baseline choice for the same reason.
Both are cheap enough to re-run from the same embeddings without re-embedding.

## 4. Expected time (RTX 4060 Ti, 16GB) — not measured on this hardware

| Step | Expected time |
|---|---|
| KMeans baseline fit (MiniBatchKMeans on ~36k x 384-dim embeddings) | seconds |
| UMAP (36k points → 5 components) | ~2-5 min, CPU-bound |
| HDBSCAN on the UMAP projection | ~1-3 min |
| c-TF-IDF labeling (both variants) | ~1 min |

No GPU step here beyond the embedding in step 2 — UMAP/HDBSCAN are CPU libraries; a GPU won't speed up this
step even with CUDA torch installed.

## 5. Assign topics + evaluate

```bash
make assign-topics          # --variant bertopic (default) -- writes the `topics` table + Prediction(task="topic") rows
make eval-topics            # scripts/generate_m7_report.py -- persists EvalRuns, renders the comparison report + model card
```

`assign-topics` defaults to the BERTopic variant (SPEC M7 names it as the deliverable; KMeans is the
CLAUDE.md-mandated comparison baseline only, never intended for deployment — see
`scripts/assign_topics.py`'s docstring). Run `--variant kmeans` first if you want to sanity-check the
pipeline against the cheaper variant before committing to BERTopic.

`eval-topics` needs both `data/embeddings/tickets_minilm_v1.parquet` (step 2) and at least one variant's
`topics.json` (step 3) — it reads flat files only, no DB dependency on what `assign-topics` wrote, and needs
neither the `topics` dependency group nor a GPU (pure pandas + `ml/evaluation/topic_metrics.py`'s NPMI, both
already default deps). Run it again any time after re-fitting to refresh the report without re-running
`assign-topics`.

## 6. Optional: LLM topic naming (SPEC M7, ~$0.20 budget)

c-TF-IDF keyword-joined labels (e.g. `"refund, order, late, delivery"`) are the committed default — this
step is a deliberate, later enhancement, off by default.

```bash
# cheap dry run first (a few cents at most):
uv run python -m ml.data.llm_topic_labels --limit 3
# then the top 30:
make topic-labels
```

Requires `OPENAI_API_KEY` and `LLM_ENABLED=true` in `.env` — refuses to spend anything otherwise. Updates
`Topic.label` in place for the top-N topics by size (excluding the outlier cluster); re-running is free
(cache hit) as long as the underlying keywords haven't changed, see the script's docstring for why this
doesn't need the "already labeled" exclusion guard `ml/data/llm_judge_summaries.py` needed.

## 7. Verify the SPEC M7 acceptance criteria

1. **≥ 30 coherent topics with labels** — `docs/m7-comparison-report.md`'s "SPEC M7 acceptance" section
   states PASS/FAIL directly; also spot-check `GET /topics` (`make up`, then `curl localhost:8000/topics`).
2. **Topics-over-time chart in dashboard** — `http://localhost:3000/topics`, reachable from the header nav.
3. **Emerging issues panel fires on an injected synthetic spike** — already proven by
   `tests/unit/test_trend_metrics.py` and `tests/integration/test_topics_api.py` in CI; optionally confirm
   the live panel too if a real topic happens to spike in the corpus's dense window.

Close the module with the CLAUDE.md acceptance-criteria checklist (criterion, evidence, eval run id) before
starting M8.
