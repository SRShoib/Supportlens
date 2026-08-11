FROM python:3.11-slim-bookworm AS base
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

WORKDIR /app

FROM base AS deps
COPY pyproject.toml uv.lock ./
# `serving` adds transformers+torch (CPU) so the API can load real M3
# transformer exports — not a GPU dep, just larger than the sklearn-only
# baseline image (see docs/decisions.md).
RUN uv sync --frozen --no-dev --no-group ml --group serving --no-install-project

FROM base AS runtime
RUN groupadd --system app && useradd --system --gid app --home-dir /app app
COPY --from=deps /app/.venv /app/.venv
COPY pyproject.toml uv.lock README.md ./
COPY apps/api ./apps/api
COPY ml ./ml
RUN uv sync --frozen --no-dev --no-group ml --group serving
ENV PATH="/app/.venv/bin:$PATH"
USER app

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
