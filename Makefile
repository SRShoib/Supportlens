COMPOSE = docker compose --env-file .env -f infra/docker-compose.yml

.PHONY: install install-training dev up down clean logs ps test test-unit test-int cov-clean lint fmt migrate revision ingest-bitext ingest-twitter build-slice build-splits seed eval tokenization-doc train-baseline-intent train-baseline-urgency seed-label-urgency

install:
	uv sync
	uv run pre-commit install
	uv run pre-commit install --hook-type pre-push

# Opt-in only (transformers/accelerate/evaluate, plus a CPU torch pulled in
# transitively). Do NOT run this expecting GPU training to work afterward —
# see docs/decisions.md for the separate CUDA torch install command.
install-training:
	uv sync --group training

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
