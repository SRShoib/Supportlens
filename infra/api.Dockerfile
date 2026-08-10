FROM python:3.11-slim-bookworm AS base
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

WORKDIR /app

FROM base AS deps
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-group ml --no-install-project

FROM base AS runtime
RUN groupadd --system app && useradd --system --gid app --home-dir /app app
COPY --from=deps /app/.venv /app/.venv
COPY pyproject.toml uv.lock README.md ./
COPY apps/api ./apps/api
RUN uv sync --frozen --no-dev --no-group ml
ENV PATH="/app/.venv/bin:$PATH"
USER app

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
