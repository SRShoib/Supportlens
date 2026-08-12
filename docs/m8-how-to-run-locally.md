# M8: running the semantic search + RAG pipeline locally

Unlike M3–M6, there's no GPU training step here — every model M8 uses is pretrained off the shelf
(`sentence-transformers/all-MiniLM-L6-v2` for embeddings, `cross-encoder/ms-marco-MiniLM-L-6-v2` for
rerank). CLAUDE.md ground rule #3 still applies to the *serving* side in spirit (no heavy model work
inside the API's request path beyond what SPEC's latency budget already assumes), but there's nothing to
fine-tune or export to `models/`.

The full pipeline below **has already been run for real** in this session against the real ingested
corpus — `docs/m8-comparison-report.md` has the real hit-rate@5 numbers and the real no-answer
demonstration, both persisted as `eval_runs` rows. The steps below are how to reproduce that, or re-run it
after the corpus changes (new tickets ingested, M5's sentiment backfill re-run, etc.).

## 0. Prerequisites

- **M1** ingestion (tickets/messages in Postgres) and **M5**'s sentiment-trajectory backfill
  (`make predict-sentiment`) must already have run — `ml/data/resolved_tickets.py`'s "resolved" definition
  reads `Prediction(task="sentiment_trajectory").score`, so a ticket with no trajectory yet can never be
  indexed, not even wrongly (see `docs/decisions.md`).
- `chroma` must be up: `make up` (or `make dev`, which starts `postgres` + `chroma` and runs the API with
  `--reload`).

## 1. One-time setup

```bash
make install-search        # uv sync --group search (sentence-transformers, chromadb==0.5.23)
```

Unlike `install-training`/`install-topics`, this group **also** ships in `infra/api.Dockerfile` — the live
`/search` and suggested-reply endpoints need it in production, not just for local scripts (see
`docs/decisions.md`'s "apps/api never loads an embedding model" entry for why M8 is different from M7 here).

## 2. Generate the synthetic KB

```bash
make kb-generate            # uv run python -m ml.data.kb_generate
```

Writes (upserts, idempotent) all 40 `KbArticle` rows to Postgres — 27 keyed to the real Bitext intent
taxonomy, 13 hand-picked from the real M7 topic catalog. No LLM call, no cost. Re-run any time after
editing `ml/data/kb_generate.py`'s `ArticleSpec` list; it won't duplicate rows.

## 3. Index resolved tickets + the KB into Chroma

```bash
make index-search           # uv run python scripts/index_search_corpus.py
```

Embeds every resolved ticket's customer-problem text (`resolution_quality > 0`, ~7,676 tickets at the real
corpus scale) and all 40 KB articles, writing both into their own Chroma collection. ~1-2 minutes on CPU.
Smoke-test first with `--limit 500` if you just want to confirm the code path. Idempotent — ticket/article
ids are stable, so re-running after new tickets are ingested only adds the new ones.

## 4. Verify the search endpoint

```bash
make up
curl -s -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "my package never arrived", "top_k": 3}' | python -m json.tool
```

Or from the dashboard: `http://localhost:3000/search`.

## 5. Build the retrieval eval set + run the report

```bash
make build-retrieval-eval   # uv run python -m ml.data.retrieval_eval_set -- writes data/eval/retrieval_queries.parquet
make eval-search            # uv run python scripts/generate_m8_report.py
```

`eval-search` needs the `search` group and a populated Chroma index (steps 1-3). It:

1. Runs all 100 eval queries through the real retrieval pipeline twice — dense-only and dense+rerank —
   computes hit-rate@5 for each, and persists both as `EvalRun` rows (`task="retrieval"`).
2. Runs 6 hand-picked queries (3 realistic, 3 clearly off-topic) through the real pipeline and checks each
   against `ml/inference/rag_reply.py`'s `MIN_CONFIDENCE` gate — SPEC M8's "no-answer behavior
   demonstrated" criterion, with real numbers regenerated every run rather than a one-off manual check.
3. Renders `docs/m8-comparison-report.md` from both.

Real numbers from this session: dense-only hit-rate@5 = 0.900, dense+rerank = 0.920 (100 queries); 6/6
no-answer examples behaved as expected.

## 6. Suggested-reply drafting (SPEC M8, part of the shared $5 project budget)

No separate script — it's the live `POST /tickets/{id}/suggested-reply` endpoint (or the "Generate
suggested reply" button on a ticket detail page in the dashboard). Requires `OPENAI_API_KEY` and
`LLM_ENABLED=true` in `.env`; refuses to spend anything otherwise (`LLMDisabledError` → 503). Every draft
is cached by `(purpose, prompt_hash)` in `llm_calls` — re-viewing the same ticket doesn't re-bill. Real
cost so far this session: **$0.0008** across every real smoke-test call combined ($0.0363 → $0.0371 total
project spend), against SPEC §5's ≈$1.50 line for this module and the project's $5 total cap.

## 7. Verify the SPEC M8 acceptance criteria

1. **Retrieval hit-rate@5 on a 100-query synthetic eval set** — `docs/m8-comparison-report.md`'s "Retrieval
   hit-rate@5" section, backed by two `eval_runs` rows (`task="retrieval"`, `model_version` `dense_v1` /
   `dense_rerank_v1`).
2. **Rerank vs no-rerank comparison** — same report's "Rerank vs no-rerank" section.
3. **RAG endpoint refuses gracefully when retrieval confidence is low** — same report's "No-answer
   behavior" section (real run, PASS), plus `tests/unit/test_rag_reply.py` and
   `tests/integration/test_rag_endpoint.py`'s dedicated refusal-path tests for regression coverage.

Close the module with the CLAUDE.md acceptance-criteria checklist (criterion, evidence, eval run id) before
starting M9.
