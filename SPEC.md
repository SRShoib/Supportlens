# SPEC.md — supportlens

**Customer Support Intelligence Platform**
An end-to-end NLP system that ingests raw customer support conversations and turns them into structured, searchable, actionable intelligence: intent and urgency classification, entity extraction, sentiment trajectories, thread summaries, emerging-issue detection, semantic search over resolved cases, and RAG-drafted reply suggestions — all served behind a FastAPI backend with a Next.js analytics dashboard.

---

## 1. Purpose & positioning

**Why this project exists.** This is a portfolio project for AI/ML engineer roles. It is designed to demonstrate the *entire* NLP task spectrum in one coherent product — from classical baselines to fine-tuned transformers to retrieval-augmented generation — plus the engineering that surrounds models: data pipelines, evaluation, drift monitoring, and deployment.

**The interview sentence it must support:**
> "I built a platform that classifies, extracts, summarizes, clusters, and searches customer support tickets — with classical baselines, fine-tuned transformers, and RAG — deployed end-to-end with FastAPI, Next.js, PostgreSQL, and a vector store."

**Design principles:**
1. **Baselines first.** Every learned component has a cheap classical baseline and a documented comparison. Knowing when a baseline is enough is the senior signal.
2. **Evaluation is a feature.** Metrics, confusion matrices, and drift detection are first-class product surfaces, not an afterthought.
3. **Budget realism.** Total paid-API spend across the whole project must stay under **$5 (OpenAI)**. Everything else runs on free Hugging Face models fine-tuned locally on the builder's own NVIDIA GPU and served on CPU in production.
4. **Real messy text.** Models are demonstrated on genuinely dirty data (real tweets/tickets), not toy sentences.

**Non-goals (v1):**
- No multi-tenant auth / user accounts (single demo workspace).
- No real-time streaming ingestion (batch + on-demand API is enough).
- No agentic orchestration (that's covered by the LangGraph portfolio projects — keep this one focused on NLP depth).
- No training inside the app containers (training happens in local scripts run directly on the builder's machine; the app only serves exported artifacts).

---

## 2. Data strategy

No single free dataset has all the labels we need, so supportlens deliberately combines sources — and documenting this strategy is itself a portfolio asset.

| Source | Role | Notes |
|---|---|---|
| **Bitext Customer Support (Hugging Face)** | Labeled intent classification (27 intents, ~27k utterances) | Clean labels for supervised intent training; synthetic but well-structured |
| **Customer Support on Twitter (Kaggle, ~3M tweets)** | Real messy text: preprocessing showcase, clustering, semantic search corpus, drift simulation | No task labels — used for unsupervised modules + domain realism. (Downloaded once from Kaggle as a dataset — unrelated to where training runs.) |
| **tweet_eval (sentiment/emotion)** | Transfer source for sentiment & emotion heads | Fine-tune once, apply to support text |
| **samsum / dialogsum** | Transfer source for dialogue summarization | Fine-tune FLAN-T5-small/base |
| **Synthetic NER set (generated)** | Token-level labels for ORDER_ID, PRODUCT, DATE, AMOUNT, ACCOUNT_REF | Template + LLM-assisted generation (~3–5k sentences), injected into real ticket shells |

**Label bootstrapping for urgency** (not present in any source): rule-based weak labels (keywords, punctuation, ALL-CAPS ratio, refund/legal terms) + a small LLM-labeled seed set (≤2,000 examples, cached, one-time cost ≈ $0.50) → train a classifier, report agreement stats between weak labels and LLM labels.

**Canonical internal schema** (everything is normalized into this):

```
Ticket { id, source, created_at, channel, customer_id, messages: [Message] }
Message { id, ticket_id, author_role: customer|agent, text, sent_at }
Prediction { ticket_id, task, label/score/entities/summary, model_version, created_at }
```

---

## 3. Architecture

Monorepo, three deployable services plus a training workspace:

```
supportlens/
├── apps/
│   ├── api/            # FastAPI — ingestion, inference, search, analytics endpoints
│   └── dashboard/      # Next.js + Tailwind — ticket queue, analytics, search UI
├── ml/
│   ├── data/           # loaders, cleaners, weak labelers, synthetic NER generator
│   ├── training/       # fine-tuning scripts (run locally on the builder's GPU, not in Docker)
│   ├── inference/      # thin CPU inference wrappers (ONNX / quantized where useful)
│   └── evaluation/     # metric computation, drift detection, LLM-judge harness
├── infra/
│   └── docker-compose.yml   # api + dashboard + postgres + chroma
├── notebooks/          # EDA + training notebooks (mirrors ml/training scripts)
└── docs/               # architecture diagram, model cards, decision log
```

**Stack (fixed — matches the builder's existing expertise):**
- Backend: **FastAPI** (Python 3.11), Pydantic v2, SQLAlchemy 2
- DB: **PostgreSQL** (tickets, messages, predictions, eval runs)
- Vector store: **Chroma** (default, persistent) with a FAISS index option for the pure-search benchmark
- Embeddings: **sentence-transformers** (e.g. `all-MiniLM-L6-v2`, upgrade candidates documented)
- Models: Hugging Face Transformers (DistilBERT/DeBERTa-v3-small, FLAN-T5-small/base), spaCy for linguistic preprocessing
- Frontend: **Next.js** (App Router) + Tailwind
- LLM (paid, capped): OpenAI small model for seed labeling, RAG reply drafting, and a tiny LLM-as-judge sample — every call cached in Postgres
- Training hardware: **local NVIDIA GPU** via a CUDA-enabled PyTorch build, run directly in VS Code (no cloud notebook dependency)
- Packaging: Docker Compose; models loaded from a local `models/` artifact directory (exported from training runs)

**Inference latency budget (CPU, single request):** classification < 150 ms, NER < 250 ms, embedding < 100 ms, summarization < 3 s (small model, greedy/beam=2). Not hard requirements, but measured and reported.

---

## 4. Modules & milestones

Each module has **acceptance criteria** — the module is done only when all boxes tick. Build order is the module order.

### M0 — Scaffold & tooling
Repo layout above; `uv` (or pip-tools) locked deps; ruff + pytest wired; pre-commit; Docker Compose skeleton (api + postgres + chroma boot and healthcheck); `make` targets (`make dev`, `make test`, `make up`); CI (GitHub Actions: lint + tests).
**Accept:** `make up` brings up all services healthy; a hello-world API test passes in CI.

### M1 — Ingestion & preprocessing pipeline
Loaders for Bitext + the Twitter dataset → canonical schema → Postgres. Cleaning pipeline: HTML/signature stripping, handle/URL/email masking, unicode normalization, language filter (keep en), emoji handling (map, don't delete), dedup. A written comparison of spaCy tokenization vs BPE subword tokenization on 20 gnarly real examples (goes in `docs/`).
**Accept:** ≥100k real messages ingested and queryable; cleaning functions ≥95% unit-test coverage; tokenization comparison doc committed.

### M2 — Intent & urgency classification (classical baseline)
TF-IDF (word + char n-grams) → Logistic Regression and Linear SVM for **intent** (Bitext 27 classes) and **urgency** (weak-label bootstrap per §2). Stratified splits, seeds fixed. Deliverable: baseline report (macro-F1, per-class F1, confusion matrix) persisted as an eval run in Postgres.
**Accept:** intent macro-F1 ≥ 0.85 on Bitext test; urgency weak-label vs LLM-seed agreement (Cohen's κ) reported; `POST /predict/intent` serves the baseline.

### M3 — Transformer fine-tuning & the comparison story
Fine-tune DistilBERT (and one stronger variant, e.g. DeBERTa-v3-small) on the same splits, **locally on the builder's NVIDIA GPU**. Export to `models/`, serve on CPU (optionally ONNX-quantized). Produce the **baseline-vs-transformer report**: accuracy delta, latency delta, size delta, and a written recommendation of which to deploy per task.
**Accept:** transformer beats baseline macro-F1 by a reported margin (or the report honestly says it doesn't and why); model card written; API flag switches baseline/transformer per request.

### M4 — Named Entity Recognition
Entities: `ORDER_ID`, `PRODUCT`, `DATE`, `AMOUNT`, `ACCOUNT_REF`. Generate the synthetic training set (templates + LLM paraphrase pass on a capped budget), inject into real ticket shells, hand-verify a 200-example gold test set. Fine-tune a HF token-classification model locally; compare against a regex/rules baseline (which will genuinely win on ORDER_ID — say so).
**Accept:** span-level F1 reported per entity on the gold set; rules-vs-model comparison table; `POST /predict/entities` live; extracted entities rendered as chips in the dashboard.

### M5 — Sentiment & emotion trajectory
Fine-tune on tweet_eval (sentiment 3-class, emotion 4-class) locally, apply per-message, aggregate per-ticket into a **trajectory** (e.g. angry → neutral → satisfied) with a simple resolution-quality heuristic (final-message sentiment × urgency).
**Accept:** eval on tweet_eval test reported; trajectory sparkline visible per ticket in dashboard; per-ticket aggregate stored as a Prediction.

### M6 — Thread summarization
Fine-tune FLAN-T5-small (or base) locally on samsum/dialogsum, apply to multi-message tickets. Evaluate with ROUGE-1/2/L on the transfer test set **plus** an LLM-as-judge pass on exactly 50 supportlens summaries (1–5 faithfulness/coverage rubric, cached, budget ≈ $0.30). Document hallucination examples found.
**Accept:** ROUGE + judge scores in an eval run; 2-line summary shown at top of every ticket view; a `docs/summarization-failure-modes.md` with ≥3 real failure examples.

### M7 — Embeddings, topic discovery & trend detection
Embed the real-ticket corpus (sentence-transformers) → BERTopic (UMAP + HDBSCAN) → human-readable topic labels (c-TF-IDF; optional one capped LLM pass to name the top 30 topics). Trend detection: topic volume per week, flag topics whose volume z-score > 2 ("emerging issues").
**Accept:** ≥ 30 coherent topics with labels; topics-over-time chart in dashboard; an "emerging issues" panel that fires on an injected synthetic spike (test fixture proves it).

### M8 — Semantic search & RAG reply suggestions
Index resolved tickets + a small synthetic KB (~40 articles) in Chroma. Search endpoint: dense retrieval → optional cross-encoder rerank (`ms-marco-MiniLM`) → results with highlighted matches. RAG "suggested reply": retrieve top-k similar resolved cases + KB articles → OpenAI drafts a reply **with citations to the retrieved sources** → agent sees draft + sources side by side. All drafts cached; hard budget guard in code (env-configured token ceiling).
**Accept:** retrieval hit-rate@5 measured on a 100-query synthetic eval set (question → known relevant ticket); rerank vs no-rerank comparison; RAG endpoint refuses gracefully when retrieval confidence is low (no-answer behavior demonstrated).

### M9 — Evaluation dashboard & drift monitoring
A `/metrics` area in the dashboard: per-model eval runs over time, confusion matrices, per-class F1, retrieval metrics, latency percentiles. Drift: embedding-distribution distance (e.g. MMD or centroid cosine shift) + prediction-distribution shift (PSI) between a reference week and the live window, with a simulated drift scenario (feed the app a topically different slice and watch the alarms fire).
**Accept:** all metrics render from Postgres eval runs (no hardcoded numbers); drift simulation documented with screenshots in `docs/`.

### M10 — Deployment, docs & portfolio packaging
One-command `docker compose up` demo with a seeded demo dataset; README with architecture diagram, GIFs, metric highlights and the data-strategy story; model cards for every fine-tuned model; a `DEMO.md` walkthrough script (5-minute path an interviewer can follow); short blog-post draft.
**Accept:** fresh-machine clone-to-running in ≤ 3 commands; README review checklist passed; demo script executed end-to-end without manual fixes.

---

## 5. Budget & compute plan

- **Paid API ceiling: $5 total.** Enforced in code by a spend-tracking wrapper around every OpenAI call (persisted counter in Postgres; hard-stop env var). Planned allocation: seed labeling ≈ $0.50, NER paraphrase pass ≈ $0.50, topic naming ≈ $0.20, LLM-judge ≈ $0.30, RAG reply drafting (demo + cache warm) ≈ $1.50, reserve ≈ $2.00.
- **Never call paid APIs from tests or CI.** All external calls mocked; cached fixtures committed for demo determinism.
- **GPU work** (M3–M6 fine-tunes) runs **locally on the builder's NVIDIA GPU**, driven directly from VS Code — no cloud notebook dependency, no session time limits. Every training script is a plain `python ml/training/train_*.py --config ...` file, runnable from the VS Code terminal or as notebook-style cells via the Jupyter extension.
- **Serving is CPU-only** — that's a feature, not a limitation: distillation/quantization decisions become part of the story.

## 6. Testing & quality bar

- Unit tests: cleaning, weak labelers, schema mappers, spend guard, chunking/indexing.
- Integration tests: API endpoints against a dockerized Postgres + Chroma (test containers), models replaced with tiny stub checkpoints.
- Eval harness (`ml/evaluation/`) is the single source of truth for every number in the README — no metric appears anywhere unless a committed eval run produced it.
- Style: ruff (lint + format), mypy on `apps/api` and `ml/inference`.

## 7. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Bitext intents too clean/synthetic → inflated scores | Report both Bitext test scores *and* qualitative performance on real tweets; discuss the domain gap explicitly |
| Weak urgency labels are noisy | Report κ vs LLM seed set; treat urgency as a "weak supervision showcase," not ground truth |
| Summarization hallucinates | Small beam, input truncation strategy documented, failure-modes doc (M6) |
| 3M tweets too heavy to process at once | Work on a stratified ~150k slice for development; scale up to more of the corpus only as local time/RAM allow |
| Budget creep | Hard-stop spend guard + cached responses committed for demo |

## 8. Definition of done (project level)

The project is portfolio-ready when: all module acceptance criteria pass; `docker compose up` + seed script gives a working demo with all nine capabilities visible in the dashboard; the README tells the baseline-vs-transformer and data-strategy stories with real numbers; and total paid spend logged < $3.
