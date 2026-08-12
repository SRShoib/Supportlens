# Decision log

Non-obvious design choices: date, decision, why, alternatives considered.

## 2026-08-10 — psycopg3 for both the API engine and bulk-ingest engine

**Decision:** use `psycopg[binary]` (v3) as the only Postgres driver, for both the FastAPI request-scoped
sessions and the M1 bulk-ingest scripts.
**Why:** one driver instead of asyncpg + psycopg2 — fewer dependencies, one connection-string dialect, one
thing to pin/upgrade. The API is not on the hot async-everything path yet (SPEC's latency budgets are
CPU-model-bound, not DB-bound), so psycopg3's sync mode is enough for M0/M1.
**Alternatives:** asyncpg (faster async, but a second driver for ingest scripts) + psycopg2 (ingest side).

## 2026-08-10 — UUIDv5 ids over random UUIDs for Ticket/Message

**Decision:** every row id is `uuid5(NAMESPACE, f"{entity_type}:{source}:{external_id}")`, where
`entity_type` is the literal `"ticket"` or `"message"`.
**Why:** re-running a loader must be a no-op, not a duplicate-row generator, so ingestion can be re-run
safely (crash recovery, schema changes, corpus updates) without a separate reconciliation step. Determinism
also means the same seed-42 run produces byte-identical ids across machines, which matters for reproducible
eval-run joins in M2+. The `entity_type` prefix was added after a real collision on the full Twitter corpus
— see the entry below.
**Alternatives:** random UUIDs + a separate `(source, external_id)` unique constraint doing the
dedup-on-conflict work — works, but loses the "same input → same id" property that later modules
(predictions keyed by ticket id, cached embeddings) benefit from.

## 2026-08-10 — Chroma boots in M0, stays unused until M8

**Decision:** `infra/docker-compose.yml` includes the `chroma` service from M0 onward, pinned to
`chromadb/chroma:0.5.23`.
**Why:** M0's acceptance criterion is literally "api + postgres + chroma boot and healthcheck" — booting it
now and only adding a client in M8 means M8 adds application code, not infrastructure risk.
**Alternatives:** add Chroma to compose only when M8 starts — rejected because it would leave the M0
acceptance criterion unmet until M8.

## 2026-08-10 — Two ingestion bugs only surfaced at real scale

**Finding:** the Twitter loader's union-find conversation grouping always produces a root_id drawn from a
real member tweet_id (a property of union-find roots, not an edge case). Without an entity-type prefix in
the id hash, `deterministic_id(source, root_id)` for the ticket and `deterministic_id(source, tweet_id)`
for that same member tweet's message produced the *same* UUID. This was already present in the small
`tests/fixtures/twcs_sample.csv` fixture (ticket "102"'s root is a real member) — the existing tests just
never asserted id-uniqueness, so nothing caught it until a live API response showed a ticket and its own
message sharing an id. At real scale (the ~150k-message Twitter slice) it hit 34,157 of 36,649 tickets
(93%). Separately, `persist_tickets` batched INSERTs by ticket count, not row count; a batch of 5000
multi-message tickets can need >65535 bind params, Postgres's hard per-statement limit — invisible below a
few hundred rows, guaranteed to fire on the real corpus.
**Why this matters:** both bugs were latent in the small-fixture test suite; only running the real data
volume surfaced them. Unit tests on tiny fixtures proved round-trip/determinism properties but not the
invariants that actually broke (id uniqueness, per-statement row limits).
**How to apply:** when a data-shape property depends on volume/structure (dedup collision rates, batch
sizes vs. DB limits), test it against something closer to real scale, or at minimum add an explicit
invariant assertion (e.g. "no id collision") rather than only checking round-trip/determinism properties.

## 2026-08-10 — models/ is a mounted volume, not baked into the API image

**Decision:** `infra/docker-compose.yml`'s `api` service mounts `../models:/app/models:ro` instead of the
Dockerfile `COPY`-ing `models/` into the image.
**Why:** M2 onward, the API needs to load real trained artifacts to serve predictions. Baking them into the
image would mean every newly trained model (M2's baseline today, M3-M6's transformers later) requires a
full image rebuild to deploy — coupling "ship new code" to "ship a new model" for no reason. A read-only
bind mount means dropping a new model into `models/` takes effect on container restart.
**Alternatives:** `COPY models ./models` in the Dockerfile — rejected for the coupling problem above, and
because it would keep growing the image size as more models accumulate through M2-M6.

## 2026-08-10 — transformers pinned below 5.0 for the training group

**Decision:** `pyproject.toml`'s `training` dependency group pins `transformers>=4.46,<5.0`.
**Why:** the open-ended `>=4.46` resolved to 5.15.0 in practice, which has a tokenizer-conversion bug that
breaks loading `microsoft/deberta-v3-small` specifically — it misroutes the model's SentencePiece file
through a tiktoken BPE-ranks parser and crashes (`ValueError: Error parsing line ... in spm.model`). Caught
by CPU-smoke-testing `ml/training/train_transformer.py` before handing it off, not by reasoning about it in
advance. The well-established 4.x series does not have this bug. Revisit the pin once upstream fixes it.
**Alternatives:** patch around it with `use_fast=False` — tried first, doesn't help, the broken conversion
path runs regardless of the fast/slow tokenizer flag.

## 2026-08-11 — a new `serving` dependency group, separate from `training`

**Decision:** `pyproject.toml` gets a `serving` group (`transformers<5.0`, `torch>=2.2`, `sentencepiece`,
`protobuf`) — excluded from `default-groups` like `training`, but synced explicitly in
`infra/api.Dockerfile` (both build stages) and in CI (`uv sync --frozen --group serving`). Unlike
`training`, torch **is** pinned directly here, and is pinned to the CPU-only wheel via a dedicated,
`explicit = true` `[[tool.uv.index]]` entry (`pytorch-cpu`, `download.pytorch.org/whl/cpu`) scoped to
`torch` only through `[tool.uv.sources]`.
**Why:** M3's accept criterion is a real API flag that serves live transformer inference (SPEC: "API flag
switches baseline/transformer per request"), not just an export step — so `ml/inference/transformer.py`
needs `transformers`+`torch` at API runtime, in Docker and in CI, not just on the training machine. **The
CPU-index pin was added after the fact, correcting this entry's original assumption that "a normal
lockfile-pinned resolution is fine" for CPU serving.** It isn't: a plain `torch>=2.2` resolution on Linux
pulls the CUDA build transitively (11 `nvidia-cu13` packages + `cuda-toolkit` + `triton`, ~2.5GB), which is
dead weight in a container with no GPU and directly violates CLAUDE.md's "never add GPU deps to the API
image" rule. This is also what caused two real local disk-space crashes (C: hit 0 bytes free, twice) during
the API image build — not a hypothetical concern. Pinning the CPU wheel drops the `serving` group's download
from ~2.6GB to ~200MB (torch alone: 502MB CUDA build → 183MB CPU-only wheel) with identical inference
correctness.
**How to apply:** never run `uv sync --group serving` (or anything that touches the `torch` pin) in the main
local dev venv used for GPU training — it silently replaces the manually-installed CUDA build documented
below with the CPU-only one (confirmed by hitting this exact trap while making this fix: `uv sync --group
serving` swapped `torch==2.6.0+cu124` for `2.13.0+cpu` locally; recovered via `uv pip install --reinstall
torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124`). Locally, the already-installed `training`
group (torch + transformers) is a superset that covers the same import surface for testing
`ml/inference/transformer.py`, so `serving` never needs to be synced outside Docker/CI — full stop, not just
"in principle."
**Alternatives:** leave the plain PyPI resolution as originally decided — rejected once it was shown to
silently resolve to the CUDA build and to be the actual root cause of the disk crashes, not just a
theoretical inefficiency; ONNX-export the transformer models instead of serving raw safetensors — still
deferred, real optimization work beyond what M3 requires.

## 2026-08-11 — the CPU-torch swap trap also fires on `--group training`, not just `--group serving`

**Finding:** the entry above's "how to apply" says the danger is `uv sync --group serving`. It isn't only
that — `uv sync --group training` hits the identical trap, confirmed by hitting it while preparing M4's CPU
smoke test: `accelerate` (a `training`-group dependency) requires `torch` without a version/build
constraint, and `[tool.uv.sources]`'s `torch = [{ index = "pytorch-cpu" }]` pin is **project-scoped, not
group-scoped** — it applies to *any* resolution of `torch`, regardless of which group's sync triggered it.
Running `uv sync --group training` here silently swapped `torch==2.6.0+cu124` for `2.13.0+cpu` again,
exactly like the `serving` case; recovered with the same command (`uv pip install --reinstall torch==2.6.0
--index-url https://download.pytorch.org/whl/cu124`), then re-verified `torch.cuda.is_available() is True`.
**Why this matters:** the previous entry's "training is a superset, so serving never needs to be synced
outside Docker/CI" is still true for *avoiding a second sync*, but it understates the risk: a `uv sync
--group training` that needs to touch the lockfile at all (a fresh checkout, a dependency bump, anything
that isn't a no-op) can just as easily strip the CUDA build.
**How to apply:** after *any* `uv sync` on a GPU training box — not just one naming `serving` — verify
`uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` still reports the
CUDA build before starting a real training run. If it doesn't, rerun the CUDA reinstall command above.

## 2026-08-10 — CUDA torch install (RTX 4060 Ti, driver 591.86)

**Decision:** torch is never a pinned dependency in `pyproject.toml` (any group). `make install-training`
pulls in a CPU-only torch transitively via `accelerate` (which hard-requires torch); the human then runs
`uv pip install torch --index-url https://download.pytorch.org/whl/cu124` once, manually, to replace it
with the CUDA build.
**Why:** CLAUDE.md requires the CUDA build to match the local driver and be a manual one-time install, not
lockfile-pinned the way CPU deps are — a pinned torch version in the shared lockfile would either force a
CPU-only build on everyone (breaking GPU training) or force a specific CUDA version that may not match
whoever's driver is running it.
**How to apply:** documented in full in `docs/m3-how-to-run-locally.md`; this entry exists so the "why not
just pin it" question has a permanent answer.

## 2026-08-12 — M5's tweet_eval splits are used verbatim, not re-split with seed 42

**Decision:** `ml/training/tweet_eval_data.py` writes tweet_eval's own `train`/`validation`/`test` partition
straight to `data/splits/{sentiment,emotion}_v1.parquet` (renaming `validation` to `val`), instead of pooling
all rows and re-splitting 70/15/15 the way `ml/training/splits.py::_stratified_split` does for M2's
Postgres-sourced data.
**Why:** tweet_eval already ships a fixed, canonical split for its own benchmark — re-splitting it would be
reinventing the benchmark, not reproducing it, and would make this repo's numbers non-comparable to published
tweet_eval results. CLAUDE.md's "seeds fixed (42) for every split" rule is about splits *we* generate from
scratch; it doesn't apply here because no random partitioning happens at all.
**Alternatives:** pool train+validation+test and re-split 70/15/15 like M2 — rejected, loses comparability to
the standard benchmark for no benefit (the fixed split is already stratified and appropriately sized).

## 2026-08-12 — Resolution-quality formula and "final message" / "ticket urgency" definitions (M5)

**Decision:** SPEC M5 specifies the heuristic only as "final-message sentiment x urgency," leaving both terms
undefined. This repo defines: **final message** = the ticket's last *customer* message (not the literal last
message, which could be the agent's closing reply); **ticket urgency** = the urgency prediction on the
ticket's *first* customer message (the same per-message unit `ml/training/splits.py::build_urgency_splits`
trained on). Formula: `resolution_quality = signed_sentiment(final_customer_message) *
URGENCY_WEIGHT[ticket_urgency]`, where `signed_sentiment` is `+score` (positive), `-score` (negative), or
`0.0` (neutral), and `URGENCY_WEIGHT = {"low": 1.0, "medium": 0.66, "high": 0.33}` — see
`ml/inference/sentiment_trajectory.py`.
**Why:** a ticket that opened urgent and ends on a negative note should score worse than one that opened
calm and ends negative; weighting by *opening* urgency (not urgency re-evaluated at the end) captures "how
much this mattered" rather than re-measuring the same negative-ending signal twice. Using the customer's last
word (not the agent's) keeps the metric about customer state, matching the trajectory's own framing.
**Alternatives:** literal last message regardless of author — rejected, would score an agent's positive
closing note as ticket resolution quality even when the customer never responded to it; re-evaluating urgency
on the final message instead of the opening one — rejected, would largely just re-derive the sentiment signal
under a different name rather than adding independent information.

## 2026-08-12 — Sentiment trajectory Predictions are full-recompute, not incrementally upserted

**Decision:** `scripts/compute_sentiment_trajectories.py` deletes every existing
`task="sentiment_trajectory"` `Prediction` row before reinserting, in the same transaction as the inserts.
**Why:** M2-M4's `/predict/*` endpoints never persist anything (always live, stateless) — M5 is the first
module writing durable `Prediction` rows, and `Prediction.id` is a random UUID (`default=uuid.uuid4`), not a
deterministic hash like `Ticket`/`Message`'s ids. Re-running the script without clearing old rows first would
just accumulate duplicates rather than being a safe no-op the way M1's ingestion loaders are.
**Alternatives:** upsert on `(ticket_id, task)` — would need a new unique constraint `Prediction` doesn't
have today (unlike `LLMCall`'s `(purpose, model, prompt_hash)` cache key) and adds schema surface for a
script that's expected to run as an occasional full backfill, not a high-frequency incremental job.

## 2026-08-12 — The canonical `samsum` HF dataset no longer loads; `knkarthick/samsum` mirror used instead

**Finding:** `datasets.load_dataset("samsum")` fails outright under this repo's `datasets>=3.1` pin —
`"trust_remote_code is not supported anymore"` — because the canonical `samsum` repo ships as a Python
loading script, and recent `datasets` versions refuse to execute loading-script datasets at all, not just
warn about them. This wasn't a hypothetical risk flagged in advance; it was caught by actually trying to
load the dataset while writing `ml/training/summarization_data.py`.
**Decision:** use `knkarthick/samsum` and `knkarthick/dialogsum` — plain CSV-backed mirrors with identical
`{id, dialogue, summary}` columns and the same split sizes as the original benchmarks (samsum:
14731/818/819, dialogsum: 12460/500/1500).
**Why:** these are the two live, no-loading-script mirrors on the Hub with matching schema and split sizes,
so they reproduce the standard benchmark rather than reinventing it — consistent with M5's tweet_eval
verbatim-split decision above.
**Alternatives:** pin an older `datasets` version that still executes loading scripts — rejected, would
conflict with the `datasets>=3.1` pin the rest of `ml/data/*` and M5 already depend on, for one dataset's
sake.

## 2026-08-12 — M6 pools samsum + dialogsum for training, evaluates ROUGE per dataset

**Decision:** `ml/training/train_summarization.py` pools both datasets' `train` (and `val`) rows into one
FLAN-T5-small fine-tune; `scripts/generate_m6_report.py` reports ROUGE-1/2/L against each dataset's own
`test` split separately, never pooled.
**Why:** samsum and dialogsum are the same task (dialogue → summary) from two different sources, not two
different label sets the way M5's sentiment/emotion split was — pooling gives the fine-tune more stylistic
diversity to learn from (dialogsum's service/call-center-style dialogues sit closer to this project's
support-ticket domain than samsum's casual messenger chats). Test-time evaluation stays per-dataset so the
numbers remain comparable to each published benchmark, the same logic as the 2026-08-12 tweet_eval entry
above — pooling test rows would produce a number that isn't comparable to anything in the literature.
**Alternatives:** train on dialogsum only (closer domain fit, but throws away samsum's larger train set);
train two separate models, one per dataset (mirrors M3's dual-variant comparison structure instead of M5's
single-model-per-task precedent) — rejected as more GPU time and report scope than the domain-gap question
actually needs.

## 2026-08-12 — M6's classical baseline is lead-k extractive summarization, k=4

**Decision:** `ml/inference/extractive_summary.py::ExtractiveSummaryPredictor` returns the ticket's first k
turns verbatim as the "summary" — no learned parameters, no model file, same shape as M4's
`RulesEntityPredictor`. `DEFAULT_K=4`, picked by sweeping k=1..6 against the real pooled samsum+dialogsum
val split (1,318 rows) and taking whichever maximizes ROUGE-1 (k=4: 0.3098 vs. k=3's 0.3095 — effectively a
tie, ROUGE-2 keeps climbing past k=6 but ROUGE-1/L both peak at k=3-4).
**Why:** SPEC M6's own accept criteria don't list a baseline comparison the way M3/M4/M5 do, but CLAUDE.md
ground rule #2 ("Never skip the classical baseline... the comparison IS the deliverable") is unconditional,
and every other learned M2-M5 component already has one. Lead-k is the standard, unglamorous extractive
summarization baseline — cheap, deterministic, and a real chance to repeat M4's "the simple thing wins
sometimes" story if FLAN-T5 underperforms on very short tickets.
**Alternatives:** TextRank or another graph-based extractive method — more sophisticated, but adds a real
dependency and tuning surface for a baseline whose entire point is being the cheap, boring comparison point;
no baseline at all — rejected, contradicts CLAUDE.md rule #2 directly.

## 2026-08-12 — Thread summaries are precomputed and persisted, not served live per ticket view

**Decision:** `scripts/compute_thread_summaries.py` backfills `task="thread_summary"` `Prediction` rows
(same full-recompute delete+reinsert shape as M5's trajectory backfill above); the dashboard's ticket page
reads the persisted row via `GET /tickets/{id}/predictions?task=thread_summary`, it never calls
`POST /predict/summary` live. The live endpoint still exists (API completeness, direct testing) and is what
the backfill script calls under the hood.
**Why:** SPEC §3's CPU latency budget allows summarization up to 3s/request — far slower than entities
(250ms) or classification (150ms) — so computing it live on every ticket-page load would make the dashboard
noticeably slower per view. M5 already established the precompute-and-persist pattern for exactly this
reason class (an aggregate that's expensive enough to not want to redo on every read).
**Alternatives:** live per-page-load via `POST /predict/summary`, mirroring M4's entities pattern — rejected
given the latency gap; M4's entities call is fast enough that live serving is a non-issue there, and that
reasoning doesn't transfer to a multi-second seq2seq generation call.

## 2026-08-12 — `uv sync` (no `--group` flag) silently uninstalled CUDA torch and the whole `training` group

**Finding:** running a bare `uv sync` (adding `rouge-score` to the default `ml` group) uninstalled
`torch==2.6.0+cu124` along with `transformers`/`accelerate`/`sentencepiece`/`tiktoken`/`psutil` outright —
not swapped to a CPU build, *removed* — because those packages were only present on this machine via a
previous `--group training` sync, and a plain `uv sync` reconciles the venv to exactly the resolved
default-groups set, uninstalling anything outside it. The two 2026-08-11 entries above document `--group
training`/`--group serving` swapping the CUDA build for a CPU one; this is the same trap's more destructive
sibling — a bare `uv sync` doesn't swap the non-default packages, it deletes them.
**How to apply:** never run a bare `uv sync` on this GPU training box once `training` has been synced here —
use `uv sync --group training` (accepting the CUDA→CPU swap that follows, per the existing recovery command)
or `uv sync --no-sync` style scoping if only touching default-group deps. After *any* `uv sync` variant here,
verify `uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` before a real
training run, exactly as the 2026-08-11 entry already prescribes.
**Recovery used:** `uv sync --group training` (restored transformers/accelerate/etc., predictably swapped
torch to `2.13.0+cpu`) → `uv pip install --reinstall torch==2.6.0 --index-url
https://download.pytorch.org/whl/cu124` (restored the CUDA build) → separately, `safetensors` was left in a
broken half-upgraded state (`ImportError: cannot import name 'TensorSpec'`, a stale compiled
`_safetensors_rust.pyd` next to newer Python wrapper code, caused by a locked file mid-upgrade — see the next
entry) → `uv pip install --reinstall safetensors` fixed it.

## 2026-08-12 — A running `uvicorn` dev server can corrupt the venv mid-`uv sync`

**Finding:** `uv sync` failed twice with `error: failed to remove file ...\_safetensors_rust.pyd: Access is
denied` while `make dev`'s uvicorn process (which has `transformers`/`safetensors` loaded in memory) was
still running. uv partially completed the install regardless (new `.dist-info` metadata written, old
compiled extension left in place), leaving `transformers` unimportable
(`ImportError: cannot import name 'TensorSpec' from 'safetensors'`) until the dev server was killed and
`safetensors` reinstalled.
**How to apply:** stop any running `uvicorn`/API dev server before `uv sync`/`uv pip install` touches
`transformers`, `torch`, or `safetensors` specifically — a locked native extension file doesn't just fail
loudly, it can leave a half-upgraded package that fails on import later, silently, at a point disconnected
from the sync command that caused it.

## 2026-08-12 — `scripts/make_stub_models.py`'s `build_all()` regenerates every fixture, not just new ones

**Finding:** adding `stub_transformer_thread_summary` and running `uv run python scripts/make_stub_models.py`
(no args) to generate it also rewrote all 8 pre-existing committed fixtures — `git status` showed them
modified even though nothing about their own recipes changed. `--check` still reported "all stub fixtures
match their reproducible recipe" immediately afterward, because at just-regenerated files trivially match
themselves; that comparison is only meaningful against the previously-committed bytes.
**How to apply:** when adding one new stub fixture, call that fixture's own `build_stub_*` function directly
(or `git checkout --` the unrelated fixtures afterward, which is what was done here) rather than running
`build_all()`/the bare CLI, to avoid an unreviewed diff across every other module's committed fixtures.

## 2026-08-12 — T5's tokenizer needs `return_token_type_ids=False`, unlike every prior stub

**Finding:** `SummarizationPredictor.predict()` crashed with `ValueError: The following model_kwargs are not
used by the model: ['token_type_ids']` when tokenizing with the generic `PreTrainedTokenizerFast`-based stub
(`stub_transformer_thread_summary`). Real T5 tokenizers never emit `token_type_ids`; a plain
`PreTrainedTokenizerFast` (built from a bare `tokenizers.Tokenizer`, the same technique
`scripts/make_stub_models.py` already used for the BERT-based stubs) does by default, and T5's `.generate()`
rejects the extra kwarg outright rather than ignoring it.
**Decision:** `ml/inference/summarization.py`'s tokenizer call passes `return_token_type_ids=False`
explicitly, rather than only fixing the stub tokenizer's construction.
**Why:** fixing it at the call site protects against *any* tokenizer that happens to emit `token_type_ids`
(stub or otherwise), not just this one stub's specific construction — a one-line, zero-cost guard versus a
fixture-specific workaround that a future different stub could reintroduce.

## 2026-08-12 — The real M6 run, and a real duplicate-judging bug found and fixed in the process

**What happened:** the real GPU fine-tune landed (`models/transformer_thread_summary_flan-t5-small_v1/final/`,
307MB, a genuine FLAN-T5-small checkpoint, not the CPU smoke-test artifact). Full-corpus transformer backfill
(~63k tickets) would have taken ~7-8 hours on CPU at the measured ~430ms/ticket, so
`scripts/compute_thread_summaries.py` gained a `--limit` flag (see the entry above this one) and was run with
`--limit 500` instead — enough to demo the feature and feed the judge sample without an overnight run.
**Bug found:** following this repo's own documented workflow (`docs/m6-how-to-run-locally.md`: "cheap dry run
first" with `--limit 5`, then the full `--limit 50`) left **5 tickets judged twice** — 55
`thread_summary_judge` rows for only 50 distinct tickets. `ml/data/llm_judge_summaries.py`'s sampling was
purely seed-deterministic with no memory of what had already been judged, so the full run's first 5 picks
were identical to the dry run's 5 (same seed, same candidate pool) — cache hits on the OpenAI call (no
re-billing) but each loop iteration unconditionally inserts a new `Prediction` row regardless of cache status.
**Fix:** `_sample_summary_predictions` now excludes tickets with an existing `thread_summary_judge` Prediction
(for the transformer model version) from the candidate pool before sampling. The 5 stale duplicate rows
(the dry run's copies, identical scores to their duplicates since the LLM call was cached) were deleted
directly from Postgres to bring the real run back to exactly 50 distinct judged tickets, per SPEC M6's
"exactly 50" wording, before `scripts/generate_m6_report.py` was re-run to persist the final EvalRun and
render the real `docs/m6-comparison-report.md`.
**Why this matters:** `ml/data/llm_judge_summaries.py` was modeled directly on `ml/data/llm_seed_labels.py`,
which has the identical unguarded-reinsert pattern (no check for "is this message already labeled" before
adding a Prediction) — this bug is latent there too, just never triggered because M2's seed-labeling was
always run as one single batch, never a small dry run followed by a larger one. Worth revisiting if
`llm_seed_labels.py` is ever re-run incrementally.

## 2026-08-12 — M7 scope: Twitter-only corpus, customer-messages-only embedding unit

**Decision:** `scripts/compute_embeddings.py` embeds `TicketSource.TWITTER` tickets only, one document per
ticket built from the concatenation of that ticket's **customer** messages (`text_clean`, chronological),
never Bitext and never agent replies.
**Why:** Bitext's `created_at` is permanently `NULL` (synthetic, single-turn, no timestamp column at all —
see the 2026-08-10 loader entries), so it can never appear on the weekly trend axis SPEC M7 also asks for;
mixing it in would give ~42% of "topics" no time dimension. Agent replies are template-heavy ("sorry to hear
that, please DM us") and would otherwise dominate every cluster's c-TF-IDF terms — the customer's own words
are what actually describe the issue, same instinct as `ml/inference/sentiment_trajectory.py`'s "final
customer message" and the urgency split's "first customer message".
**Alternatives:** embedding the full thread (rejected — agent boilerplate pollutes topic terms); embedding
both sources with Bitext tickets bucketed into an "unknown" trend week (rejected — makes the weekly chart
misleading for no real benefit, since Bitext was never meant to represent real-world timing anyway).

## 2026-08-12 — M7 gets a classical baseline SPEC's text doesn't ask for

**Decision:** `ml/training/topic_model.py` fits a TF-IDF-labeled MiniBatchKMeans baseline (fixed
`kmeans_n_clusters`, no outlier cluster) alongside BERTopic, from the same embeddings and c-TF-IDF labeling
step, and `scripts/generate_m7_report.py` compares them on NPMI coherence.
**Why:** SPEC M7's module text names only BERTopic, but CLAUDE.md ground rule #2 ("baselines before
transformers, always — the comparison IS the deliverable") and SPEC §1 principle #1 apply to every learned
component, and every other module (M2–M6) shipped one. Skipping it here would make M7 the only module
without a baseline-vs-learned story. sklearn is already a default dependency, so this costs no new packages.
**Alternatives:** LDA instead of KMeans (more conventional for topic modeling, but slower to fit on ~36k docs
and typically scores worse on coherence); no baseline at all, following SPEC's literal wording (rejected —
breaks the project-wide rule for a one-off exception SPEC never actually argued for).

## 2026-08-12 — M7 "coherent" made measurable: NPMI, no hardcoded threshold

**Decision:** `ml/evaluation/topic_metrics.py` computes NPMI (normalized pointwise mutual information,
Lau/Newman/Baldwin 2014) over each topic's top-10 c-TF-IDF terms, from the corpus's own document
co-occurrence counts — no gensim, no reference corpus. `scripts/generate_m7_report.py` reports mean NPMI per
variant as evidence; the SPEC acceptance bar itself stays literal (≥ 30 topics, HDBSCAN's `-1` outlier
cluster never counted toward it).
**Why:** CLAUDE.md rule #5 ("no metric without an eval run") means "coherent" has to become a number, but
SPEC M7 never defines one or a pass/fail threshold — inventing a threshold SPEC never set would be a bigger
overreach than reporting the metric and leaving the bar exactly as written.
**Alternatives:** a hand-picked NPMI threshold for "coherent" (rejected — arbitrary, no SPEC basis); UMass
coherence (rejected — needs a reference corpus's document frequencies at a specific window size, more
machinery for a portfolio project than NPMI's plain co-occurrence-count version).

## 2026-08-12 — M7 trend detection: a dense analysis window, and z-scores on volume *share*

**Decision:** `ml/evaluation/trend_metrics.py`'s `select_dense_window` first drops week buckets under 10% of
the median non-empty week's volume and keeps only the longest contiguous run; `compute_topic_trends` then
computes each topic's weekly *share* of total volume (not raw count) and flags `share z-score > 2` with a
leave-one-out mean/stdev (excludes the week being scored, so one huge spike can't inflate its own stdev and
suppress its own z-score) plus two hard gates: `MIN_HISTORY_WEEKS = 4` (skip, don't flag, on thin history)
and `MIN_SPIKE_TICKETS = 5` (blocks the `0,0,0,1`-style degenerate case where a flat-zero history's stdev is
effectively 0 and one single ticket would otherwise register an enormous z-score).
**Why:** the real twcs corpus spans 2008-05 to 2017-12 but is 99.6% concentrated in ~10 weeks of late 2017 —
naive `date_trunc('week', ...)` over the whole range produces ~480 near-empty buckets that make any z-score
meaningless. Raw-count z-scores are also confounded by total-volume swings: if every topic's ticket count
rises together (the whole queue got busier), that's not one topic "emerging" — scoring on share isolates the
signal SPEC M7 actually asks for.
**Alternatives:** a fixed calendar window (rejected — arbitrary, doesn't adapt to where the corpus's data
actually is); raw-count z-scores (rejected — the global-volume confound above); population stdev instead of
leave-one-out (rejected — lets a spike's own week suppress its own z-score, the opposite of what "flag the
spike" needs).

## 2026-08-12 — M7 embeddings stay a local artifact; the `topics` dependency group is offline-only

**Decision:** `scripts/compute_embeddings.py` writes `data/embeddings/tickets_minilm_v1.{npy,parquet}`
(gitignored, like `data/splits/`) — no Chroma client is added in M7. `sentence-transformers`, `bertopic`,
`umap-learn`, and `hdbscan` all live in one new `topics` dependency group, excluded from `default-groups`
exactly like `training`; every module that imports them (`ml/inference/embeddings.py`,
`ml/training/topic_model.py`'s `fit_bertopic`) is lazily imported from inside the scripts that need it
(`scripts/compute_embeddings.py`, `ml/training/topic_model.py`), never at a module top level that
`apps/api` or a default `pytest` run could reach.
**Why:** the 2026-08-10 "Chroma boots in M0, stays unused until M8" entry already commits to Chroma's
client/application code landing in M8, not earlier; M7 pulling it forward would contradict that decision for
no real benefit, since M8's collection shape (message-level vs ticket-level, resolved-only filtering, KB
articles too) is different enough from M7's needs that little would actually be reused. Keeping `apps/api`
and CI's default `uv sync` free of BERTopic/UMAP/HDBSCAN/sentence-transformers also means
`ml/evaluation/trend_metrics.py` and `topic_metrics.py` — the two modules the acceptance-critical "fires on
an injected spike" test targets — stay testable without that group installed at all.
**Alternatives:** write embeddings to Chroma now so M8 only adds search/rerank (rejected — pulls the
vector-store integration risk forward into M7, and contradicts the committed decision-log entry above).

## 2026-08-12 — A `psutil` install in this venv was silently a stub, broke MiniBatchKMeans

**Finding:** `sklearn.cluster.MiniBatchKMeans.fit_predict` crashed with `AttributeError: module 'psutil' has
no attribute 'Process'` (raised deep inside joblib's loky backend, `_cpu_count_affinity`) while unit-testing
`ml/training/topic_model.py`'s KMeans baseline. `uv pip show psutil` reported version 7.2.2 installed, but
`import psutil; psutil.__file__` was `None` — an empty namespace package, not the real library.
**Fix:** `uv pip install --reinstall psutil` — uv reported `Failed to uninstall package at
...psutil-7.2.2.dist-info due to missing RECORD file`, confirming the prior install was already corrupted,
not merely stale. Same failure class as the safetensors/torch corruption entries above (a partially-written
package left behind by an earlier `uv sync`/reinstall cycle on this machine), just surfacing in a different
package this time.
**Why this matters:** `psutil` isn't a direct dependency of anything in `pyproject.toml` — it's pulled in
transitively (`accelerate`, per the 2026-08-11 entries), so a broken copy is invisible until something deep
in a dependency's dependency actually calls into it. Worth a quick `uv pip show <pkg>` + import sanity check
on this machine after any `uv sync` if an unrelated-looking `AttributeError`/`ImportError` shows up mid-run.

## 2026-08-12 — The real M7 run, and two real bugs found and fixed in the process

**What happened:** the real pipeline landed on the full ~36.6k-ticket Twitter slice —
`make embed-tickets` → `ml/training/topic_model.py` → `scripts/assign_topics.py` →
`scripts/generate_m7_report.py`, all on CPU. BERTopic discovered 54 topics (10,132-ticket outlier
cluster) against SPEC M7's ≥ 30 bar; KMeans's fixed `kmeans_n_clusters=40`.

**Bug 1 — masking-token artifacts and unfiltered stopwords in c-TF-IDF keywords:** the first render
of `docs/m7-comparison-report.md` showed top topics like `"user, url, the, in"` and
`"user, emoji, url, face_with_tears_of_joy"` — not human-readable labels (SPEC M7's explicit
requirement). Root cause: a `TfidfVectorizer`/`CountVectorizer`'s default tokenizer strips
punctuation before counting, collapsing `ml/data/masking.py`'s `<USER>`/`<URL>`/`<EMAIL>`/`<PHONE>`
tokens into bare `"user"`/`"url"`/... — words that then alias with organic English usage and are
near-universal across the corpus (almost every ticket mentions the brand's @handle or a URL), so
instead of being suppressed the way a true common-to-every-cluster term should be, they dominated
nearly every topic. Separately, `fit_bertopic`'s `BERTopic(...)` call never configured a
`vectorizer_model`, so — unlike the KMeans baseline's `_ctfidf_keywords`, which already passed
`stop_words="english"` — BERTopic's own labels had no stopword filtering at all (`"the"`, `"in"`,
`"you"`, `"of"` shown as top terms).
**Fix:** `ml/training/topic_model.py:TOPIC_STOP_WORDS` — sklearn's English stopword list unioned with
the mask tokens' bracket-stripped, lowercased forms (derived from `MaskToken` directly, not
hardcoded, so it can't drift from `ml/data/masking.py`) — passed to both vectorizers. Real effect:
BERTopic's top topic went from `"user, url, the, in"` (5,347 tickets) to `"food, store, just,
chicken"`; mean NPMI rose from 0.1687 to 0.2259 (KMeans: 0.1072 → 0.1425). Covered by a regression
test (`tests/unit/test_topic_model.py::test_ctfidf_keywords_filters_masking_tokens_and_english_stopwords`)
built from realistic masked-text input.
**Bug 2 — chart x-axis label overlap:** `topics-over-time-chart.tsx` rendered one `<text>` per week,
built and browser-verified against a synthetic 6-week fixture during development. The real corpus's
dense analysis window (`select_dense_window`, data-driven, not a fixed guess) turned out to be 78
weeks — every label overlapped into unreadable noise. Fixed by capping to at most 9 evenly-spaced
labels regardless of week count; re-verified in a browser against the real 78-week dataset, light and
dark.
**Also cleaned up:** re-running `scripts/generate_m7_report.py` after the c-TF-IDF fix left 2 stale
pre-fix `EvalRun` rows (task="topics", the 0.1072/0.1687 NPMI numbers) alongside the 2 correct ones —
`persist_eval_run` always inserts, it doesn't upsert, same as every other M-report script. Deleted the
stale pair directly from Postgres, same remediation as the M6 duplicate-judging entry above.
**Why this matters:** both bugs were invisible against synthetic test fixtures — the masking-token
collapse only shows up with real masked text at real corpus frequency, and the label overlap only
shows up with the real corpus's actual (larger, data-driven) week count. Neither is a gap in the unit
tests being wrong, exactly; it's the general lesson M2–M6 already priced in (SPEC §7: "qualitative
performance on real tweets" sections exist for exactly this reason) — a synthetic fixture proves the
logic is correct, not that it's tuned for what real data actually looks like.

## 2026-08-12 — M8 "resolved tickets" defined from M5's resolution_quality, not "has an agent reply"

**Decision:** `ml/data/resolved_tickets.py::resolved_ticket_ids` defines a resolved ticket as a
`TicketSource.TWITTER` ticket whose `sentiment_trajectory` Prediction (SPEC M5,
`ml/inference/sentiment_trajectory.py`) has `Prediction.score` (resolution_quality) strictly positive.
**Why:** the canonical schema has no explicit resolution status anywhere (no `status` column, no loader
ever sets one) and SPEC M8's text ("index resolved tickets") doesn't define one either. The first candidate
— "has at least one agent message" — was measured against the real ingested corpus before being written:
36,578 of 36,579 Twitter tickets have an agent reply (99.997%). The twcs dataset's conversation grouping is
curated around brand-response threads, so nearly every captured ticket includes one by construction; that
definition doesn't discriminate anything, so indexing "resolved tickets" under it would mean indexing
essentially the whole Twitter corpus, defeating the point of a "similar *resolved* cases" retrieval corpus
for RAG. resolution_quality > 0 does discriminate (7,677 of 36,579 Twitter tickets at the real-run scale,
~21%) and is semantically closer to what the RAG use case needs — "show the agent an example of this kind
of issue being resolved *well*" — since M5 already ties it to the customer's final-message sentiment
discounted by opening urgency. Bitext is excluded entirely, same reasoning as every real-corpus module
since M7: synthetic single-turn instruction/response pairs, no real resolution to speak of.
**Alternatives:** `resolution_quality >= 0` (rejected — includes exact-neutral endings, ~59% of the corpus,
too permissive to read as "resolved well"); "agent had the last message" (rejected — untested against the
real corpus and likely close to as degenerate as the agent-reply-exists candidate, for the same structural
reason); a brand-new heuristic independent of M5 (rejected — M5 already computed and persisted a signal
that means almost exactly what's needed here, inventing a second one would duplicate reasoning already
priced into `sentiment_trajectory.py`'s design, see its module docstring).

## 2026-08-12 — M8 KB articles: templated, not LLM-generated; 13 of 40 hand-picked from real M7 topics

**Decision:** `ml/data/kb_generate.py` generates all 40 SPEC-M8 KB articles from hardcoded, hand-authored
`ArticleSpec` templates — 27 keyed to the real Bitext intent taxonomy (queried from Postgres:
`Ticket.meta["intent"]` across all ingested `TicketSource.BITEXT` rows, confirmed exactly 27 distinct
values) and 13 keyed to hand-picked entries from the real M7 `topics` table
(`model_version="topics_bertopic_v1"`), rewritten as clean brand-agnostic prose rather than rendered
mechanically from their raw c-TF-IDF keyword strings.
**Why:** SPEC §5's M8 budget line ("RAG reply drafting (demo + cache warm) ≈ $1.50") is earmarked for reply
drafting specifically; there's no separate line for writing the KB, so an LLM-generated KB would either
overrun the module's own budget allocation or eat into the shared reserve for no clearly-scoped reason.
Rendering the real topic catalog directly was tried conceptually and rejected once the actual topic labels
were inspected: several of the real clusters are noise ("wtf, does, jpg, love"; "fuck, worst, suck, hate")
or too vague to summarize ("service, customer, hold, chat") — a KB "article" auto-titled from one of those
would read as garbage or, worse, profanity, in a portfolio demo. The 13 chosen topics are all real,
coherent clusters (flights, gaming account issues, ride-share trips, train tickets, streaming playback,
mobile OS updates, internet outages, package delivery, baggage, card payments, software updates, TV/channel
access, food/store orders) — each article's `source_key`/`tags` still cite the originating `topic_key` for
traceability, but the prose itself is hand-written, not templated from the keyword list.
**Alternatives:** LLM-drafted articles from intent/topic names (rejected — budget, plus SPEC's own explicit
LLM-judge/RAG-drafting budget lines suggest LLM calls are meant to be scoped narrowly, not used as a
default tool everywhere convenient); mechanical rendering of all real M7 topics including outliers/noise
clusters (rejected — quality, see above); Bitext-only (27 articles, short of "~40" and loses the
real-corpus grounding story that's part of this project's overall data-strategy narrative, SPEC §2).

## 2026-08-12 — M8 RAG confidence gate: cross-encoder score > 0, measured against the real corpus

**Decision:** `ml/inference/rag_reply.py::MIN_CONFIDENCE = 0.0`, checked against the top retrieved source's
**cross-encoder** score (`cross-encoder/ms-marco-MiniLM-L-6-v2`, always applied inside `draft_reply` —
independent of whatever `rerank` flag a `/search` UI call used) — below it, the endpoint refuses before ever
calling the LLM (SPEC M8's "no-answer behavior").
**Why:** measured directly against the real indexed corpus (7,676 resolved tickets + 40 KB articles) rather
than guessed: 5 realistic support queries ("my package never arrived", "how do I reset my password", "I was
charged twice for my order", "flight got cancelled and I need a refund", "my internet has been down for 2
days") scored their best source at cross-encoder logits `[0.93, 9.02]`; 5 clearly off-topic queries (trivia,
small talk, gibberish — "what is the capital of France", "purple elephants dance under the moonlight", etc.)
scored `[-11.17, -3.33]` — a wide, clean gap either side of 0. Raw dense cosine similarity was checked too
and rejected as the gate signal: on-topic queries scored similarity 0.60–0.71, off-topic scored 0.28–0.53 —
close enough (e.g. one off-topic query at 0.526 vs. an on-topic query at 0.604) that a clean threshold isn't
obvious the way it is for the cross-encoder score.
**Alternatives:** gating on dense similarity instead (rejected — the overlap above); a threshold picked
without measurement (rejected — CLAUDE.md rule #5's spirit: don't invent an unjustified number when the
real corpus is sitting right there to check against, same instinct as M7's NPMI/z-score decisions).

## 2026-08-12 — Bug found by real smoke test: a ticket's own text retrieved itself as a RAG source

**Finding:** the first real (non-mocked) call to `POST /tickets/{id}/suggested-reply` — for a ticket about a
password/iPhone sign-in issue — returned a draft citing 4 "similar resolved cases", but source `[1]` was the
*exact same ticket* being drafted for. Root cause: a ticket's Chroma id in the `resolved_tickets` collection
is its own `ticket_id` (`scripts/index_search_corpus.py`), and `suggested_reply` queries with that same
ticket's own customer-problem text — querying with a document's own exact text against its own exact
embedding is close to a perfect match, so it dominated the pool every time. Unit tests never caught this:
every fixture used distinct ticket ids for "the ticket" and "the candidate sources", a distinction the real
data doesn't preserve because there's exactly one document per ticket, and the query text passed to
`draft_reply` always came from the very ticket being drafted for.
**Fix:** `ml/inference/retrieval.py::retrieve` gained an `exclude_ids: frozenset[str]` parameter (default
empty, so `apps/api/routers/search.py` — which never has anything to exclude — is unaffected);
`ml/inference/rag_reply.py::draft_reply` takes `exclude_ticket_id` and passes it through;
`apps/api/routers/rag.py` passes `str(ticket_id)`. Re-verified directly against the real corpus (bypassing
the LLM call to avoid re-billing for a retrieval-only check): the same ticket's best self-excluded source
score dropped from a self-match to genuinely different tickets, and a second real end-to-end call (different
ticket, ~$0.0003) confirmed the drafted reply's sources no longer include the ticket itself.
**Why this matters:** same lesson as M7's real-run entries above — a synthetic fixture proves the retrieval
*logic* is correct, not that it's exercised the way real data actually shapes it (here: query text and one
candidate document being derived from the exact same source object). Worth remembering for any future
retrieval feature keyed by a document's own content.

## 2026-08-12 — M8 Chroma integration tests: real server via testcontainers, pinned to the deployed version

**Decision:** `tests/integration/conftest.py` adds `chroma_container` (session-scoped
`testcontainers.community.chroma.ChromaContainer`, pinned to `chromadb/chroma:0.5.23` — testcontainers'
own default is `1.0.0`, a different heartbeat API version than what `infra/docker-compose.yml` actually
runs) and `chroma_store` (a real `ChromaVectorStore` against it, resetting the two router-facing collection
names before every test — Chroma has no `TRUNCATE`, so this is delete-and-recreate instead). Both are
gated behind `pytest.importorskip("chromadb")` since `chromadb` lives in the `search` group, not CI's
default `--group serving` sync — every test using them skips cleanly rather than failing when it's absent,
verified directly: re-ran the full integration suite after `uv sync --frozen --group serving` (no `search`)
and got 8 clean skips, zero failures, matching what CI's job actually has installed.
**Why:** CLAUDE.md: "Integration tests use testcontainers for Postgres/Chroma" — every other M8 test
(search router, RAG drafting) injects a fake store, which never exercises
`ml/inference/vector_store.py`'s actual `chromadb.HttpClient` wiring at all. Real testing paid off
immediately: the real server rejected `metadatas=[{}]` (empty dict) with `"Expected metadata to be a
non-empty dict"` — a constraint the fake `FakeCollection` test double happily accepts and
`scripts/index_search_corpus.py` never violates in practice (every real metadata dict has several keys),
but would have been invisible without a real server in the loop.
**Alternatives:** keep every M8 Chroma test on the fake store (rejected — leaves the actual HTTP client
integration completely unverified, the same gap CLAUDE.md's testcontainers rule exists to close); use
testcontainers' default `chromadb/chroma:1.0.0` image (rejected — validates a server version this project
doesn't deploy).

## 2026-08-12 — M8 embedding unit for tickets: customer-problem text, full thread in metadata

**Decision:** `resolved_tickets`' embedded/matched text (`document`, what `/search`'s cosine similarity and
highlighting both operate on) is the same customer-messages-only concatenation M7 already used
(`scripts/compute_embeddings.py::build_documents`, reused directly). The full thread — customer *and*
agent, in order — rides along in `metadata["thread_text"]` and is never embedded or matched against.
**Why:** the two features need different things from the same ticket. Retrieval matches a live query
("my package never arrived") against what a customer *asked*, so the indexed text should read like a
question, not a resolved conversation — embedding the full thread would mean a ticket's own agent reply
("we've refunded you") pollutes what the query is being matched against. RAG drafting is the opposite: it
needs the agent's resolution specifically to draft from, which customer-only text doesn't have. Storing
both once, keyed by the same ticket, avoids either feature quietly regressing the other.
**Alternatives:** embed the full thread (rejected — agent boilerplate/resolution text would dominate match
scoring, e.g. many resolved tickets share near-identical "please DM us" agent replies that would otherwise
cluster unrelated issues together); two separate Chroma collections per ticket, one per embedding unit
(rejected — doubles indexing/storage for no benefit once metadata already carries the second view for free).

## 2026-08-12 — M8 retrieval eval queries: first customer message, tickets with ≥ 2 required

**Decision:** `ml/data/retrieval_eval_set.py` samples 100 resolved tickets (seed 42) restricted to those
with **at least 2** customer messages, and uses the **first** one as the query text — never the ticket's
own full indexed document.
**Why:** the indexed document a query needs to *find* is the concatenation of every customer message on
that ticket (see the embedding-unit entry above). A ticket with only 1 customer message has a query
identical to its own indexed text — a free, meaningless "hit". Requiring ≥ 2 guarantees every eval query is
a genuine partial view of a longer document it needs to actually retrieve, not a lookup of itself.
**Alternatives:** allow single-message tickets too (rejected — inflates hit-rate@5 with trivial identical-
text lookups, undermining the metric's honesty); paraphrase the query text with an LLM for more realistic
phrasing (rejected — budget: SPEC M8's line item is for reply drafting, not eval-set construction, and the
zero-cost approach is already non-trivial per the ≥2-messages requirement above).

## 2026-08-12 — M8 breaks the "apps/api never loads an embedding model" precedent, on purpose

**Decision:** `infra/api.Dockerfile` syncs a new `search` dependency group (sentence-transformers +
chromadb) alongside `serving`, and `apps/api/routers/search.py`/`rag.py` load a real embedder (and,
depending on the request, a real cross-encoder) at request time.
**Why:** the 2026-08-10 "Chroma boots in M0, stays unused until M8" entry already committed to this landing
in M8; the specific mechanism is worth recording because it reverses `ml/inference/embeddings.py`'s own
M7-era docstring claim that apps/api never loads an embedding model. That claim was true for M7 (the whole
corpus is embedded once, offline, and only *assignments* are read back live) but can't hold for M8: dense
retrieval on an arbitrary live query has no offline equivalent — there is no "precomputed embedding" for a
question nobody has asked yet. Every model load stays lazily imported (`ml/inference/embeddings.py`'s
docstring, `apps/api/routers/search.py`'s `_get_*` functions) so a deployment that never hits `/search` or
the RAG endpoint never pays the import cost, but the *dependency*, unlike `topics`, has to ship in the image.
**Alternatives:** none seriously considered — a live semantic search endpoint structurally requires this;
the only real choice was making the precedent-break explicit here rather than leaving the M7 docstring's
claim silently wrong.

## 2026-08-12 — M8 budget guard: a token ceiling on top of the dollar ceiling, both required

**Decision:** `ml/inference/llm_client.py::LLMClient.complete()` gained an optional `max_tokens` parameter
(previously absent — every prior caller got an unbounded completion), and `ml/inference/rag_reply.py`
always passes `settings.rag_max_completion_tokens` (env `RAG_MAX_COMPLETION_TOKENS`, default 400).
**Why:** SPEC M8 explicitly asks for "a hard budget guard in code (env-configured token ceiling)" — distinct
wording from the dollar-based `LLM_BUDGET_USD` guard `LLMClient` already had. The dollar guard stops *future*
calls once the running total crosses a line; it does nothing to bound any single call's own completion
length, which a token ceiling does directly and which also keeps typical suggested-reply drafts from running
needlessly long (a support reply longer than ~400 tokens is itself a smell, not just a cost one).
**Alternatives:** treat the existing dollar guard as satisfying SPEC's wording (rejected — different failure
mode, and SPEC's phrase "token ceiling" is specific enough that reinterpreting it would be silently
reinterpreting an acceptance criterion, which CLAUDE.md's ground rule #1 says to raise, not do quietly).

## 2026-08-12 — M9 latency EvalRuns get their own script, not folded into M3-M6's report scripts

**Decision:** `scripts/generate_m9_latency_report.py` is a new, standalone script that loads every
already-exported predictor across M2-M7 (17 targets: intent/urgency baseline+2 transformer variants each,
entities rules+2 token-classification variants+the hybrid router, sentiment/emotion baseline+transformer,
thread_summary extractive+FLAN-T5, M7's sentence embedder) and persists one `EvalRun` per target
(`split="latency"`, `dataset="latency_probe"`), rather than adding a `persist_eval_run` call to M3-M6's
existing report scripts at the point each of them already computes a `LatencyResult` via
`ml/evaluation/latency.py::benchmark_latency` (confirmed by reading each script: M3/M4/M5/M6 all already
benchmark latency, none of them ever persisted it — only printed it or wrote it into that module's own
markdown report and model cards).
**Why:** SPEC M9's accept criterion ("all metrics render from Postgres eval runs") requires latency
percentiles to exist in `eval_runs` at all, which they didn't until now. Reusing M3-M6's own latency
computation would mean re-running those scripts' full test-set accuracy evaluation (thousands of
predictions, GPU-fine-tuned transformer inference over a full split in some cases) just to get a latency
number, and — since `persist_eval_run` has no upsert (see the M6/M7 duplicate-row entries above) — would
duplicate every accuracy `EvalRun` row on every latency-only rerun. A standalone script only loads each
model once and times a fixed probe text per task.
**Bug found and fixed before the real run landed:** the first draft reused the generic `_transformer()`
helper (which loads `TransformerPredictor`, i.e. `AutoModelForSequenceClassification`) for M4's entity
models too. Those are token-classification checkpoints; loading them through the sequence-classification
head produced a real `transformers` warning ("newly initialized" pooler weights) and would have timed the
wrong architecture. Fixed with a dedicated `_token_classification()` helper using
`TokenClassificationPredictor`; the two stale entities rows from the first run were deleted from Postgres
and the script re-run clean, same remediation pattern as the M6/M7 entries above.
**model_version naming:** every row's `model_version` matches the string its task's own accuracy `EvalRun`
already uses (M3/M5's `transformer_{model_slug}_v1`, M4's `transformer_entities_{model_slug}_v1`), so a
latency row and its accuracy counterpart group under the same `(task, model_version)` pair in
`GET /eval-runs`. Two new identifiers introduced here: `hybrid_ner_v1` for the rules+model router
`apps/api/routers/predict.py` actually serves at `model="transformer"` (distinct from the pure
token-classification model's own `EvalRun`, since it's a different predictor with a different latency
profile), and `all-MiniLM-L6-v2` for M7's sentence embedder, which has no accuracy `EvalRun` at all (topic
coherence is scored on the fitted topic model, not the embedder).
**Real run result:** all 17 targets benchmarked clean, every model under its SPEC §3 p50 budget on this
machine (`docs/m9-latency-report.md`) — worth re-running after any future retrain, since these numbers are
machine-specific and will drift with hardware.
**Alternatives:** add the persist call directly to M3-M6's scripts (rejected, cost above); benchmark only
the "deployed" model per task instead of every variant (rejected — M3-M6's own reports already benchmark
every variant, not just the winner, and the comparison value is the same here).

## 2026-08-12 — M9 drift: reference week / live window choice, data sources, and threshold calibration

**Decision:** `scripts/compute_drift.py` computes both of SPEC M9's drift signals — centroid cosine shift
(embedding-distribution distance) and PSI (prediction-distribution shift, over the urgency label) — between
a single **reference week** (`2017-10-09`) and a 4-week **live window** (`2017-11-06` through `2017-11-27`),
both drawn from the real Twitter corpus's stable high-volume tail (~3500-3600 tickets/week each), separated
by a 2-week gap so they aren't adjacent. Each signal also runs a **simulated** scenario: the same reference
week against a slice of Bitext tickets (~3000 for the embedding signal, all 26,872 persisted for the PSI
signal) standing in for "topically different live traffic" (SPEC M9: "feed the app a topically different
slice"). Embedding vectors reuse M7's already-computed corpus embeddings
(`data/embeddings/tickets_minilm_v1.{npy,parquet}`) for every real-corpus week; the Bitext slice is embedded
once, offline, in this script (Bitext was deliberately excluded from M7's own embeddings — see the M7
embedding-scope entry above). Urgency labels for both scenarios are read straight from M5's already-persisted
`sentiment_trajectory` Prediction payloads (`payload["urgency_label"]`) — zero new inference for either
signal.
**Why these specific weeks:** queried directly against the real ingested corpus (36,579 Twitter tickets,
`created_at` spanning 2008-05 to 2017-12). `select_dense_window` (M7's own dense-window logic, reused as-is)
returns a 146-week contiguous run, but volume inside it ramps from 2 tickets/week (2015-02) to ~3600/week
(late 2017) — the sparse early weeks would make either signal's centroid/PSI estimate noise-dominated.
`2017-10-09` through `2017-11-27` is the last 8 weeks of that run, all within a stable 3500-3600 range —
picked from *actually querying* the real weekly counts, not guessed.
**Why urgency (not intent) for the prediction-drift signal:** already decided in the M9 planning discussion —
intent is Bitext-trained and Bitext's `created_at` is always NULL (2026-08-12 M7 embedding-scope entry), so
it structurally can't be bucketed by week at all. Urgency and sentiment both could; urgency was picked as the
more operationally meaningful signal for a support-ticket system.
**Threshold calibration (measured against the real corpus, same methodology as M7's z-score and M8's
MIN_CONFIDENCE):**
- PSI uses the standard, already-established interpretation bands (0.1 watch / 0.25 alarm) — not
  corpus-specific, so not re-derived, but confirmed against real data anyway: real week-vs-week urgency PSI
  measured 0.0064 (reference week vs. the real live window); the simulated Bitext injection measured 0.6679
  — both land squarely inside their expected band.
- Centroid cosine shift has no external convention, so `EMBEDDING_DRIFT_THRESHOLD = 0.05`
  (`ml/evaluation/drift_metrics.py`) was picked from a direct measurement: real week-vs-week shift measured
  0.0028-0.0044 across several week pairs tried during development; the simulated Bitext injection measured
  0.60-0.64. The threshold sits roughly an order of magnitude above the real-noise ceiling and well below the
  simulated signal, the same wide-clean-gap shape M8's confidence-gate measurement found.
**Real run result:** both signals PASS SPEC M9's acceptance framing — real reference-week-vs-live-window
comparisons fire no alarm on either signal (cosine shift 0.0028, PSI 0.0064, both "stable"/no-alarm), the
simulated Bitext-injected scenario fires both (cosine shift 0.6044 alarm, PSI 0.6679 alarm) — see
`docs/m9-drift-report.md`, generated from the persisted `eval_runs` rows, not hand-typed.
**Alternatives:** recompute embeddings live for whatever window is requested instead of reusing M7's static
artifact (rejected in the M9 planning discussion — duplicates M7's artifact and pulls a live-embedding
dependency into a batch eval script for no real benefit); gate the simulated scenario behind a full M1
ingestion pipeline run instead of an eval-script comparison (rejected in the same discussion — heavier,
duplicates M1 machinery for a one-off demo, and M7/M8 already established the "eval-script simulation,
screenshot the rendered result" pattern for exactly this kind of acceptance evidence).

## 2026-08-12 — M9 API layer: GET /eval-runs is a thin read, GET /drift has its own fixed-shape schema

**Decision:** `apps/api/routers/eval_runs.py` exposes `GET /eval-runs` (optional `task`/`model_version`
filters, `limit`, newest-first) using the `EvalRunOut` schema that was already sitting in
`apps/api/schemas/eval_run.py` unused since it was scaffolded — no computation happens in the router, it's a
direct `select(EvalRun)`. `apps/api/routers/drift.py` exposes `GET /drift` returning a dedicated
`DriftOut { real: {embedding, prediction}, simulated: {embedding, prediction} }` shape (each leaf is the
latest `EvalRun` for that `(task, split)` pair, or `null` if `scripts/compute_drift.py` hasn't run yet)
rather than making the dashboard filter a generic `/eval-runs?task=drift_embedding` list client-side into
the 2x2 shape it actually needs.
**Why:** `GET /eval-runs` needs to stay generic (every task's runs, arbitrary filters) since the `/metrics`
page's per-task sections all read from it; the drift panel specifically needs a fixed 4-cell shape (real vs.
simulated x embedding vs. prediction) every time, which is worth a small dedicated endpoint+schema rather
than repeating the same "pick latest per (task,split) pair" grouping logic in the frontend.
**Alternatives:** fold `/drift` into `/eval-runs` with extra query params (rejected — the 2x2 shape doesn't
map cleanly onto a flat list endpoint's response type); compute drift live in the router instead of reading
persisted `EvalRun` rows (rejected — contradicts SPEC M9's own accept criterion, "all metrics render from
Postgres eval runs", and every other M9/M7/M8 read endpoint already follows the same "API only reads
durably-stored eval runs/predictions" contract).

## 2026-08-12 — M9 `/metrics` dashboard: one bulk fetch, generic + task-specific components

**Decision:** `apps/dashboard/src/app/metrics/page.tsx` makes exactly two requests
(`listEvalRuns({ limit: 500 })` + `getDrift()`), groups the ~90-row `eval_runs` result by `task` in the
Server Component, and renders SPEC M9's five named areas as their own components: `confusion-matrix.tsx`
(hand-rolled HTML-table heatmap, not SVG — this is tabular data, and a table lets assistive tech read real
cell values) + `per-class-f1-bars.tsx` for the four classification tasks (intent/urgency/sentiment/emotion),
`span-metrics-table.tsx` for M4's per-entity-type F1 (span metrics have no confusion-matrix equivalent —
spans aren't a fixed-size grid), `retrieval-panel.tsx` for M8's dense-vs-rerank hit-rate@5, `latency-table.tsx`
for the newly-persisted latency EvalRuns (SPEC §3 budget flagged per row), and `drift-panel.tsx` for the 2x2
real/simulated × embedding/prediction grid. A generic `eval-runs-table.tsx` (caller-supplied metric columns)
covers everything else (per-task "all runs" comparison lists, topics NPMI, summarization ROUGE + judge
scores) rather than a dedicated component per remaining task.
**Why:** SPEC M9 names five metric surfaces explicitly ("per-model eval runs over time, confusion matrices,
per-class F1, retrieval metrics, latency percentiles"); those get first-class components. Everything else
(entities' span-level per-type isn't a confusion matrix; topics/summarization/judge have no chart type SPEC
calls out) gets the generic table rather than inventing bespoke visualizations SPEC never asked for. Fetching
once and grouping client-side (well, server-side in the RSC) avoids ~10 separate task-filtered API calls for
a page whose entire dataset is currently under 100 rows.
**Color:** dataviz-skill jobs, matching this dashboard's own established conventions rather than introducing
new ones — magnitude (confusion matrix, per-class F1, retrieval bars) = single sequential blue hue, the same
hue `topics-over-time-chart.tsx` already uses as its first categorical slot (different chart context, no
legend collision); state (latency OK/OVER, drift stable/watch/alarm) = the status palette, reusing
`emerging-issues-panel.tsx`'s existing red "alarm" class exactly and `sentiment-sparkline.tsx`'s emerald
"positive"/stable convention, plus amber for PSI's middle "watch" band. Every status badge ships an icon +
text label (never color alone); every confusion-matrix/span-table cell shows its raw number as text, color
is a reinforcing channel only.
**A real bug caught by browser-verifying against real data, not by `tsc`/`eslint` (both passed cleanly):**
the first draft's Summarization ROUGE table read `runsByTask.get("thread_summary")` unfiltered, which now
also contains the new `split="latency"` rows the M9 latency script persists under the same `task="thread_summary"`
— those rows have no `rouge1`/`rouge2`/`rougeL` keys, so `.toFixed()` on `undefined` 500'd the whole page.
Fixed by filtering `split !== "latency"` before the ROUGE table renders (`isAccuracyRun`), the same filter
already applied to the classification and entities sections. Re-verified via a headless Playwright
screenshot pass (light + dark, `console --errors` clean, real Postgres data) — see
`docs/screenshots/m9-drift-real-vs-simulated-{light,dark}.png` for the drift panel specifically.
**Alternatives:** a client-rendered page with per-section `useEffect` fetches (rejected — every other
dashboard page in this app is a plain async Server Component, no reason for `/metrics` to be the exception,
and RSC data is already server-fetched once at request time); SVG for the confusion matrix instead of an
HTML table (rejected — the skill's own principle plus M7's topics-over-time-chart.tsx precedent both favor
SVG for continuous/positional charts, but a matrix is inherently tabular data with a real column/row
structure a `<table>` expresses directly).
