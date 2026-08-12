<title>DEMO</title>

# DEMO.md — a 5-minute walkthrough

A script an interviewer (or you, before an interview) can follow start to finish. Assumes `make demo` has
already run (see [README.md](README.md#quickstart)) — the dashboard is at http://localhost:3000, the API at
http://localhost:8000/docs.

Every step below hits real, seeded data — nothing is mocked or hardcoded. The seed is a curated ~339-ticket
slice of the real corpus this project was actually built and evaluated on (27 Bitext intents, a real dense
week of Twitter support traffic); see `docs/decisions.md` for exactly how it was chosen.

## 1. Classification + entity extraction (~1 min)

Open **Tickets** → click any Bitext ticket. You'll see:
- The ticket's intent, predicted live by the classical baseline (`POST /predict/intent`) — Bitext's 27
  intents, near-ceiling accuracy on this synthetic-but-structured dataset (`docs/m2-baseline-report.md`).
- Open a Twitter ticket instead and you'll see urgency (low/medium/high) predicted the same way.
- Entity chips (order IDs, products, dates, amounts, account refs) highlighted inline — a regex/rules
  extractor by default, live, no model file needed at all.

**The story:** every classifier here has a documented baseline-vs-transformer comparison with real numbers
(`docs/m3-comparison-report.md`) — intent's transformer actually *loses* to the baseline (reported honestly,
not cherry-picked); urgency's transformer wins by +0.115 macro-F1. The demo serves baselines live by default
(zero-setup); training a transformer locally and dropping its export into `models/` switches the live
`model=transformer` flag on with no code change.

## 2. Sentiment trajectory + thread summary (~1 min)

Open a Twitter ticket with more than one message. You'll see:
- A sentiment sparkline across the whole thread (customer + agent), ending in a resolution-quality score.
- A 2-line thread summary at the top (FLAN-T5-small, beats a lead-k extractive baseline by +0.16-0.17
  ROUGE-1 — `docs/m6-comparison-report.md`).

Both are precomputed (not live per-request — SPEC's own latency budget for summarization is 3s, too slow
for every page load) and shipped with the seed.

## 3. Topics, trends & emerging issues (~1 min)

Open **Topics**. You'll see:
- The real topic catalog: 54 topics discovered by BERTopic across the full corpus (mean NPMI 0.226 vs a
  0.143 TF-IDF/KMeans baseline — `docs/m7-comparison-report.md`), every label human-readable.
- A volume-over-time chart and an **emerging issues** panel that's actually firing: a real spike (topic
  "package, delivery, delivered, packages", z-score > 2) was deliberately preserved at reduced scale when
  the seed was curated, specifically so this panel demonstrates its real detection logic rather than always
  showing an empty state.

## 4. Semantic search + RAG suggested replies (~1 min)

Open **Search**, type something like *"my package never arrived"* or *"how do I reset my password"*. You'll
see dense-retrieval results (optionally cross-encoder reranked — reranking wins hit-rate@5 by
0.900 → 0.920 on a 100-query eval set, `docs/m8-comparison-report.md`) with highlighted matches across
resolved tickets and a 40-article synthetic KB.

Open a resolved ticket and click **Suggested reply** for a RAG-drafted, cited response — or a clearly
off-topic query to see the endpoint refuse gracefully rather than hallucinate (SPEC M8's "no-answer
behavior", demonstrated for real in `docs/m8-comparison-report.md`).

## 5. The eval dashboard & drift monitoring (~1 min)

Open **Metrics**. Every number on this page — confusion matrices, per-class F1, retrieval hit-rate,
CPU latency percentiles against SPEC's own budgets, ROUGE, topic coherence — renders from a Postgres
`eval_runs` row, not a hardcoded figure (CLAUDE.md rule #5).

Scroll to **Drift monitoring**: two scenarios side by side. "Real" (this week vs. a few weeks back) shows
no alarm on either signal — normal traffic isn't drift. "Simulated" (the same reference week against an
injected, topically-different slice) fires both alarms cleanly — the demonstration SPEC M9 asks for, with
real, measured thresholds (`docs/decisions.md`).

## What you just saw

Nine capabilities, one coherent platform, every number traceable to a committed eval run: classification
(baseline + transformer, compared honestly), entity extraction (rules + transformer, compared honestly),
sentiment/emotion trajectories, thread summarization, topic discovery + trend/emerging-issue detection,
semantic search + reranking, RAG-drafted replies with citations and graceful refusal, and an evaluation +
drift-monitoring dashboard treating "the numbers" as a first-class product surface rather than an
afterthought.

Total paid LLM spend across the entire project so far: **under $0.04** of a $5 hard-capped budget
(`ml/inference/llm_client.py`; the live count is always visible via the `llm_calls` table).
