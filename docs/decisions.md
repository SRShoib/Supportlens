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

**Decision:** every row id is `uuid5(NAMESPACE, f"{source}:{external_id}")`.
**Why:** re-running a loader must be a no-op, not a duplicate-row generator, so ingestion can be re-run
safely (crash recovery, schema changes, corpus updates) without a separate reconciliation step. Determinism
also means the same seed-42 run produces byte-identical ids across machines, which matters for reproducible
eval-run joins in M2+.
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
