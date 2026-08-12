COMPOSE = docker compose --env-file .env -f infra/docker-compose.yml

.PHONY: install install-training install-topics install-search dev up down clean logs ps test test-unit test-int cov-clean lint fmt migrate revision ingest-bitext ingest-twitter build-slice build-splits seed eval eval-transformers tokenization-doc train-baseline-intent train-baseline-urgency seed-label-urgency ner-data ner-paraphrase ner-gold-export ner-gold-import train-ner eval-ner sentiment-emotion-data train-baseline-sentiment train-baseline-emotion predict-sentiment eval-sentiment summarization-data train-summarization predict-summary judge-summaries eval-summarization embed-tickets fit-topics assign-topics topic-labels eval-topics kb-generate index-search build-retrieval-eval eval-search

install:
	uv sync
	uv run pre-commit install
	uv run pre-commit install --hook-type pre-push

# Opt-in only (transformers/accelerate/evaluate, plus a CPU torch pulled in
# transitively). Do NOT run this expecting GPU training to work afterward —
# see docs/decisions.md for the separate CUDA torch install command.
install-training:
	uv sync --group training

# Opt-in only (sentence-transformers/bertopic/umap-learn/hdbscan for M7's
# offline topic pipeline, plus a CPU torch pulled in transitively). Same
# CUDA-swap caveat as install-training — see docs/decisions.md.
install-topics:
	uv sync --group topics

# sentence-transformers + chromadb for M8's live search/RAG endpoints. Also
# synced into infra/api.Dockerfile (unlike install-topics) — see
# docs/decisions.md.
install-search:
	uv sync --group search

dev:
	$(COMPOSE) up -d --wait postgres chroma
	uv run uvicorn api.main:app --reload --app-dir apps/api --port 8000

up:
	$(COMPOSE) up -d --wait

down:
	$(COMPOSE) down

clean:
	$(COMPOSE) down -v

logs:
	$(COMPOSE) logs -f

ps:
	$(COMPOSE) ps

test:
	uv run pytest

test-unit:
	uv run pytest tests/unit

test-int:
	uv run pytest tests/integration -m integration

cov-clean:
	uv run pytest tests/unit --cov=ml.data.cleaning --cov=ml.data.masking --cov=ml.data.dedup --cov-fail-under=95

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy apps/api ml/inference ml/data ml/evaluation

fmt:
	uv run ruff check --fix .
	uv run ruff format .

migrate:
	uv run alembic upgrade head

revision:
	uv run alembic revision --autogenerate -m "$(m)"

ingest-bitext:
	uv run python -m ml.data.cli ingest-bitext

ingest-twitter:
	uv run python -m ml.data.cli ingest-twitter

build-slice:
	uv run python -m ml.data.cli build-slice

seed:
	uv run python -m ml.data.cli seed

eval:
	uv run python scripts/generate_baseline_report.py

eval-transformers:
	uv run python scripts/generate_m3_report.py

tokenization-doc:
	uv run python scripts/compare_tokenization.py

build-splits:
	uv run python -m ml.training.splits

train-baseline-intent:
	uv run python -m ml.training.train_baseline_intent

train-baseline-urgency:
	uv run python -m ml.training.train_baseline_urgency

seed-label-urgency:
	uv run python -m ml.data.llm_seed_labels

ner-data:
	uv run python -m ml.data.ner.generate

ner-paraphrase:
	uv run python -m ml.data.ner.paraphrase

ner-gold-export:
	uv run python scripts/ner_gold_export.py

ner-gold-import:
	uv run python scripts/ner_gold_import.py

train-ner:
	uv run python ml/training/train_token_classification.py --config ml/training/configs/ner/distilbert_cased.yaml

eval-ner:
	uv run python scripts/generate_m4_report.py

sentiment-emotion-data:
	uv run python -m ml.training.tweet_eval_data

train-baseline-sentiment:
	uv run python -m ml.training.train_baseline_sentiment

train-baseline-emotion:
	uv run python -m ml.training.train_baseline_emotion

predict-sentiment:
	uv run python scripts/compute_sentiment_trajectories.py

eval-sentiment:
	uv run python scripts/generate_m5_report.py

summarization-data:
	uv run python -m ml.training.summarization_data

train-summarization:
	uv run python ml/training/train_summarization.py --config ml/training/configs/thread_summary_flan_t5_small.yaml

predict-summary:
	uv run python scripts/compute_thread_summaries.py

judge-summaries:
	uv run python -m ml.data.llm_judge_summaries

eval-summarization:
	uv run python scripts/generate_m6_report.py

embed-tickets:
	uv run python scripts/compute_embeddings.py

fit-topics:
	uv run python ml/training/topic_model.py --config ml/training/configs/topics_minilm_bertopic.yaml

assign-topics:
	uv run python scripts/assign_topics.py

topic-labels:
	uv run python -m ml.data.llm_topic_labels

eval-topics:
	uv run python scripts/generate_m7_report.py

kb-generate:
	uv run python -m ml.data.kb_generate

index-search:
	uv run python scripts/index_search_corpus.py

build-retrieval-eval:
	uv run python -m ml.data.retrieval_eval_set

eval-search:
	uv run python scripts/generate_m8_report.py
