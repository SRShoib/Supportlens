COMPOSE = docker compose --env-file .env -f infra/docker-compose.yml

.PHONY: install dev up down clean logs ps test test-unit test-int cov-clean lint fmt migrate revision ingest-bitext ingest-twitter build-slice seed eval tokenization-doc

install:
	uv sync --all-groups
	uv run pre-commit install
	uv run pre-commit install --hook-type pre-push

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
	uv run mypy apps/api ml/inference ml/data

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
	@echo "eval harness lands in M2 (ml/evaluation)"

tokenization-doc:
	uv run python scripts/compare_tokenization.py
