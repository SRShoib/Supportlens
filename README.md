# supportlens

Customer Support Intelligence Platform — see [SPEC.md](SPEC.md) for the full product spec and module
roadmap, and [CLAUDE.md](CLAUDE.md) for working conventions.

**Status:** M0 (scaffold & tooling) in progress.

## Quickstart

```bash
cp .env.example .env
make install   # uv sync + pre-commit hooks
make up        # docker compose: api + postgres + chroma, waits for healthy
make test      # unit + integration tests
```

`make up` requires Docker Desktop running. `make` itself requires GNU Make
(`winget install ezwinports.make` on Windows).

## Layout

```
apps/api/     FastAPI service — ingestion, inference, search, analytics endpoints
ml/           loaders, cleaners, training scripts, inference wrappers, evaluation harness
infra/        Docker Compose + Dockerfiles
alembic/      database migrations
tests/        unit/ (no Docker) and integration/ (testcontainers)
docs/         decision log, model cards, tokenization comparison
```

See `docs/decisions.md` for the non-obvious design choices behind this layout.
