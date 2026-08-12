#!/bin/sh
# Container start sequence for apps/api. Split out of the Dockerfile's CMD
# (which used to be a single inline `sh -c "alembic ... && uvicorn ..."`)
# so a free-tier deploy target with no persistent disk (e.g. Render's free
# web services -- see docs/decisions.md) can opt into re-seeding the small
# git-committed demo dataset on every boot, without changing behavior for
# local `docker compose up` / `make dev`, which never sets these env vars.
set -e

alembic upgrade head

# Opt-in only (unset for local docker-compose). ml/data/seed_demo.py is
# idempotent for the Postgres tables (ON CONFLICT DO NOTHING against
# stable, pre-assigned ids) and unconditionally rebuilds both Chroma
# collections from whatever's now in Postgres -- the part that actually
# matters here when CHROMA_EMBEDDED_PATH points at ephemeral container
# disk that gets wiped on every restart/redeploy.
if [ "$RUN_DEMO_SEED_ON_BOOT" = "true" ]; then
  python -m ml.data.seed_demo
fi

# Render (and most PaaS free tiers) assign the listen port dynamically via
# $PORT and require the process to bind to exactly that port -- 8000 stays
# the default so local docker-compose (which maps the port itself) is
# unaffected.
exec uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-8000}"
