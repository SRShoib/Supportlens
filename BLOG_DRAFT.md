<title>BLOG_DRAFT</title>

# Building supportlens: a customer-support NLP platform, end to end

*Draft — not yet published.*

## Why I built this

Most NLP portfolio projects pick one task and go deep: a sentiment classifier, a summarizer, a RAG demo.
That's a reasonable way to learn a technique, but it doesn't answer the question an interviewer actually
has: *can this person take an NLP system from raw, messy data to something running in production, making
the right calls at every layer along the way?*

So I built supportlens to cover the whole spectrum on one real problem — customer support tickets —
end to end: classical baselines, fine-tuned transformers, retrieval-augmented generation, and the
unglamorous engineering that has to surround all of it (data pipelines, evaluation, drift monitoring,
deployment) for any of it to mean something in practice.

## The rule that shaped every decision: baselines first

The project's one non-negotiable rule was that every learned component ships with a classical baseline and
an honest, numeric comparison — never "the transformer will obviously win, skip the baseline." That
discipline paid off in the least glamorous way possible: on Bitext's intent classification task, a TF-IDF +
LinearSVC baseline (3.4MB, 1.6ms p50 latency) actually *beat* a fine-tuned DistilBERT (256MB, 11ms) by a
hair. Reporting that honestly — and keeping the baseline as the deployed default — is a more interesting
signal to show an interviewer than a wall of green transformer wins would have been. It's also just true:
Bitext's 27 intents are synthetic, templated instructions, and a linear classifier is already enough to
separate them near-perfectly. The domain-gap story (synthetic Bitext vs. messy real tweets) shows up
concretely every time a module gets evaluated on both.

Where the transformer *did* win, the margins were real and worth the cost: urgency classification (a
DeBERTa-v3-small fine-tune) improved macro-F1 by +0.115 over the baseline; sentiment and emotion transfer
from tweet_eval improved by +0.064 and +0.106 respectively; FLAN-T5-small beat a lead-k extractive
summarization baseline by 0.16-0.17 ROUGE-1. Entity extraction went the other way again — a boring
regex/rules extractor beat a fine-tuned token-classification model on 4 of 5 entity types, losing only on
free-form amounts. The deployed system is a hybrid: rules for order IDs, products, dates, and account
references; the transformer for amounts — decided by the actual gold-set numbers, not intuition about which
approach "should" win.

## Data strategy as its own deliverable

No single free dataset has the labels this project needs, so it deliberately combines five: Bitext for
clean intent labels, a 3M-tweet real customer-support corpus for everything that needs messy real text
(preprocessing, clustering, semantic search, drift simulation), tweet_eval as a transfer source for
sentiment/emotion, samsum/dialogsum for dialogue summarization, and a synthetic-but-real-shell-injected
named-entity dataset with a 200-example hand-verified gold set. Urgency has no source dataset at all — it's
bootstrapped entirely from weak labels (keyword/punctuation heuristics) plus a small LLM-labeled seed set,
a deliberate weak-supervision showcase rather than something the project pretends is ground truth.

Documenting *why* each source was chosen, and where its limitations show up in the numbers, turned out to
be as valuable a portfolio artifact as any individual model.

## Evaluation and drift as product surfaces, not afterthoughts

Every number in this project's README, model cards, and dashboard comes from a persisted evaluation run in
Postgres — there's a hard rule against hardcoding a metric anywhere. That sounds like process for its own
sake until you're three modules in and realize you can regenerate every report from scratch, cross-check
numbers against each other, and catch real bugs (a duplicate-row eval-run bug, a masking-token collapse in
topic keyword extraction, a self-retrieval bug in the RAG pipeline) simply because the numbers are queryable
rather than typed into a markdown file once and forgotten.

The same discipline extends to drift monitoring: rather than asserting that embedding-distribution distance
and prediction-distribution shift detectors *would* work, the project measures a real reference-week-vs-
live-window comparison on the actual corpus (stays quiet — normal week-to-week traffic isn't drift) and a
simulated topically-different injection (fires cleanly on both signals). Both thresholds were calibrated
against real measurements, not guessed round numbers.

## What actually breaks when you run the real thing

The most consistently useful lesson across ten modules: synthetic test fixtures prove logic is *correct*,
never that it's *tuned for what real data looks like*. A rules-based ID-collision bug only showed up at the
real corpus's scale (93% of tickets, invisible in a 5-row test fixture). A topic-labeling bug (masking
tokens like `<USER>`/`<URL>` collapsing into generic English words and dominating every cluster) only showed
up once real masked text hit real frequency. And the final module — packaging this for a one-command
demo — surfaced the same lesson one more time: the full `docker compose up` stack had never actually been
run end-to-end before, only approximated via a host-side dev workflow. Doing that for real found four
genuine bugs (an unmigrated database, a container-networking misconfiguration, a module that silently
dragged a training-only dependency into the serving image, and a missing writable cache for a model
download) that no amount of reasoning about the Docker config in the abstract would have caught.

## What's next

A hosted, always-on demo instance is the natural next step, along with ONNX-quantizing the CPU-served
transformers (already a documented option, not yet exercised) and pushing the entity gold set past 200
examples to tighten the per-entity confidence intervals that are currently wide enough to call several
per-entity deltas noise rather than signal. The comparison-report structure that shaped every module here
makes both easy to slot in without disturbing anything already shipped.
