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
