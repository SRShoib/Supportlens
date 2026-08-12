FROM python:3.11-slim-bookworm AS base
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

WORKDIR /app

FROM base AS deps
COPY pyproject.toml uv.lock ./
# `serving` adds transformers+torch (CPU) so the API can load real M3
# transformer exports; `search` adds sentence-transformers+chromadb (CPU) so
# it can embed a live query, rerank, and talk to the chroma service (SPEC
# M8 — see docs/decisions.md for why this breaks M7's "apps/api never loads
# an embedding model" precedent).
RUN uv sync --frozen --no-dev --no-group ml --group serving --group search --no-install-project

FROM base AS runtime
RUN groupadd --system app && useradd --system --gid app --home-dir /app app
COPY --from=deps /app/.venv /app/.venv
COPY pyproject.toml uv.lock README.md ./
COPY apps/api ./apps/api
COPY ml ./ml
COPY scripts ./scripts
COPY alembic ./alembic
COPY alembic.ini ./alembic.ini
COPY data/seed ./data/seed
RUN uv sync --frozen --no-dev --no-group ml --group serving --group search
ENV PATH="/app/.venv/bin:$PATH"
# .env's HF_HOME (./data/raw/hf, a *host*-side path for local dev) is
# overridden to a fixed in-image location by docker-compose.yml -- baked
# (not left to a first-request download) because the container runs as the
# non-root `app` user with no write access under /app, and M8's live
# search/RAG endpoints and M10's demo-seed indexing step both need
# sentence-transformers/cross-encoder weights at request/seed time (M10
# finding: the api container had never actually been run this way before,
# only via `make dev` against the host's own HF cache -- see
# docs/decisions.md).
ENV HF_HOME=/app/.cache/huggingface
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')" \
    && chown -R app:app /app/.cache
USER app

EXPOSE 8000
# Migrating on container start (idempotent -- a no-op once the schema is
# already at head) is what makes `docker compose up` alone leave a usable
# dev/demo database (M10: "fresh-machine clone-to-running"). Previously
# nothing in the documented quickstart ever migrated the dev Postgres --
# only tests/integration's testcontainers fixture did, against a disposable
# DB (docs/decisions.md).
CMD ["sh", "-c", "alembic upgrade head && uvicorn api.main:app --host 0.0.0.0 --port 8000"]
