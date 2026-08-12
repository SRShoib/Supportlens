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
RUN uv sync --frozen --no-dev --no-group ml --group serving --group search
ENV PATH="/app/.venv/bin:$PATH"
USER app

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
