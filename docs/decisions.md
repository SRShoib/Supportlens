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
