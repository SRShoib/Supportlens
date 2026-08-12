# Deploying supportlens for free

**Dashboard: Vercel.** **Backend (Postgres + Chroma + API): Railway.**

This repo has no GitHub remote yet (checked via `git remote -v` — empty),
so step 0 covers getting it there. Everything after that assumes it's on
GitHub, since both Railway and Vercel deploy by watching a connected repo.

## 0. Push this repo to GitHub

1. Go to [github.com/new](https://github.com/new). Pick a name, choose
   public or private (either works for both Railway and Vercel — private
   is fine, they authenticate via your GitHub account either way). **Do
   not** check "Add a README", "Add .gitignore", or "Choose a license" —
   this repo already has all of those; adding them on GitHub's side would
   create conflicting files with nothing to merge against yet.
2. GitHub shows you a remote URL after creating it
   (`https://github.com/<you>/<repo>.git`). Back in this repo, from the
   repo root:

   ```
   git status
   ```

   Review what's about to be committed (everything from this session:
   the dashboard redesign, `railway.json`, `render.yaml`, `DEPLOY.md`,
   the entrypoint script, the keep-warm workflow, etc. — nothing under
   `models/`, `data/raw/`, or `.env` gets picked up, those stay
   gitignored).

3. Stage, commit, and connect the remote:

   ```
   git add -A
   git commit -m "feat: dashboard redesign + Railway/Vercel deployment prep"
   git remote add origin https://github.com/<you>/<repo>.git
   git push -u origin master
   ```

   (This repo's current branch is `master`, not `main` — that's fine,
   both Railway and Vercel let you pick whichever branch to deploy from
   during setup. Renaming it is optional and not required for anything
   below to work.)

4. Confirm: refresh the GitHub repo page, the files should all be there.

This only serves the classical **baseline** models live (already the
default everywhere in this codebase). The 6.1GB of fine-tuned transformer
weights aren't in git (`.gitignore` excludes them on purpose) and won't fit
free-tier RAM regardless of host — the transformer-vs-baseline comparison
stays visible on `/metrics` as historical eval-run numbers and in
`docs/model-cards/`, exactly as already persisted, just not re-run live.
Every endpoint already degrades gracefully when a transformer model is
missing (a pre-existing 503, not something added for this) — including
entity highlighting on ticket pages, which falls back to rules-based
extraction automatically (`src/lib/api.ts`'s `predictEntities`).

## Cost reality, stated plainly

**Railway is not free past a one-time $5 / 30-day trial.** After that, it
needs the Hobby plan: **$5/month minimum, plus metered usage on top**
(CPU/memory/egress/volume storage) for whatever the three services
(Postgres, Chroma, API) actually consume while running continuously —
Railway services stay on 24/7 by default (no forced sleep the way some
other platforms' free tiers have), so this will be an ongoing real cost,
likely in the ballpark of $5-15/month depending on traffic; check Railway's
own usage dashboard once things are running for a real number rather than
trust an estimate here. Vercel's Hobby plan, by contrast, is genuinely free
indefinitely for the dashboard.

If you want a genuinely $0/month backend instead, this project also has a
tested Render-based path (Postgres via Neon, Chroma running embedded) —
see "Alternative: fully free on Render" at the bottom of this file. That
setup exists and works; this Railway-first version of the guide is what
you asked for instead.

## Why Railway changes the architecture (vs. the Render path)

Railway supports real **persistent volumes** even during the trial, and
services don't sleep by default — two things Render's free tier doesn't
offer. That means on Railway there's no need for the workarounds the
Render path required:

- **Postgres**: Railway's own managed Postgres plugin, not Neon. No
  expiry, backed by a real volume.
- **Chroma**: a real, separate, always-on Chroma service (the same
  `chromadb/chroma:0.5.23` image `docker-compose.yml` already uses
  locally) with a persistent volume — not the embedded-on-disk fallback
  the Render path needs. `CHROMA_EMBEDDED_PATH` stays unset; the app uses
  its normal networked `CHROMA_HOST`/`CHROMA_PORT` path, same as local
  dev.
- **No keep-warm workflow needed** — Railway doesn't force services to
  sleep on idle (that's an opt-in "Serverless" toggle you should leave
  off), so the cold-start-timeout-collision problem the Render path has
  with Vercel doesn't apply here.

`RUN_DEMO_SEED_ON_BOOT=true` is still worth setting even though Railway's
disk persists — the seed script is idempotent, so it's a harmless few
extra seconds on each restart in exchange for one less manual step.

---

## 1. Railway — create the project and the three services

1. Sign up at [railway.com](https://railway.com) and connect your GitHub
   account. No credit card needed for the trial.
2. **New Project → Empty Project.**

### 1a. Postgres

3. Inside the project: **New → Database → Add PostgreSQL.** Railway
   provisions it automatically — nothing else to configure.
4. Open the Postgres service's **Variables** tab and copy its
   `DATABASE_URL` value (looks like
   `postgresql://postgres:pass@postgres.railway.internal:5432/railway`).
   **Change the scheme** from `postgresql://` to `postgresql+psycopg://`
   — `apps/api/db/session.py` passes this straight to SQLAlchemy, which
   needs the driver named explicitly (this project uses `psycopg` v3).
   Save the rewritten string for step 1c.

### 1b. Chroma

5. **New → Empty Service.** Name it exactly `chroma` (later steps assume
   this name).
6. In its Settings, set the **Docker Image** to `chromadb/chroma:0.5.23`
   (under Source, instead of connecting a GitHub repo).
7. **Settings → Volumes → New Volume**, mount path `/chroma/chroma`
   (matches `docker-compose.yml`'s own volume mount — this is what makes
   the index survive restarts).
8. **Settings → Networking**: do *not* generate a public domain for this
   service — the API reaches it privately. Confirm/set its internal port
   to `8000`.

### 1c. The API

9. **New → GitHub Repo**, select this repository. Railway detects
   `railway.json` at the repo root automatically (`infra/api.Dockerfile`,
   healthcheck at `/healthz`).
10. **Variables** tab, add:

    | Key | Value |
    |---|---|
    | `DATABASE_URL` | the rewritten `postgresql+psycopg://...` string from step 4 |
    | `CHROMA_HOST` | `${{chroma.RAILWAY_PRIVATE_DOMAIN}}` (or literally `chroma.railway.internal` if the reference doesn't resolve) |
    | `CHROMA_PORT` | `8000` |
    | `RUN_DEMO_SEED_ON_BOOT` | `true` |
    | `ENV` | `production` |
    | `LOG_LEVEL` | `INFO` |
    | `LLM_ENABLED` | `false` |
    | `LLM_BUDGET_USD` | `5.00` |
    | `RAG_MAX_COMPLETION_TOKENS` | `400` |
    | `OPENAI_API_KEY` | leave blank for now — see the RAG note below |

    Leave `CHROMA_EMBEDDED_PATH` unset entirely (this is the Render-only
    fallback; setting it here would skip the real Chroma service you just
    built).
11. **Settings → Networking → Generate Domain** — this is the one service
    that needs to be public. Note the resulting URL, e.g.
    `https://supportlens-api-production.up.railway.app`.
12. Deploy. First build takes a while (installs torch/transformers/
    sentence-transformers/chromadb — the heaviest part of the whole
    setup).
13. Verify: `curl https://<your-railway-url>/healthz` should return
    `{"status":"ok","db":"ok",...}`.

**About the RAG / suggested-reply feature:** optional, same trade-off as
any other host — leaving `OPENAI_API_KEY` blank and `LLM_ENABLED=false`
deploys everything else fully working. This URL is public once deployed,
so anyone who finds it could exhaust the $5 budget guard before you get to
demo it — consider only enabling it right before you need it.

## 2. Vercel (dashboard)

1. Sign up at [vercel.com](https://vercel.com), connect GitHub, **Add New
   > Project**, select this repo.
2. Set **Root Directory** to `apps/dashboard` (this is a monorepo). Vercel
   should auto-detect Next.js and pnpm from there.
3. Add environment variable `API_BASE_URL` = your Railway API URL from
   step 1c.11 (no trailing slash).
4. Deploy.

## 3. Verify end to end

Visit the Vercel URL. Check, in order: **Overview** loads with real stat
tiles, **Tickets** lists real tickets, a **ticket detail** page shows
summary/sentiment/entities, **Topics** shows the volume chart, **Search**
returns results, **Metrics** shows real eval-run numbers, **Try it live**
returns baseline predictions (transformer column correctly shows "not
available" — expected, already-handled degradation, not a bug).

---

## Alternative: fully free on Render (no Railway cost, ever)

This project also supports deploying the API on Render's free tier
instead, using Neon for Postgres and an embedded (on-disk, rebuilt on
every boot) Chroma instead of a separate service — Render's free web
services have no persistent disk at all, which is what makes that
different from the Railway setup above. Genuinely $0/month indefinitely,
with two trade-offs: Render's free tier sleeps after 15 minutes idle
(~30-60s cold start), and it only has 512MB RAM (already the reason
transformers aren't live on *any* of these free/cheap paths).

1. **Neon** ([neon.tech](https://neon.tech)) — free, permanent Postgres.
   Create a project, copy the connection string, rewrite
   `postgresql://` to `postgresql+psycopg://`.
2. **Render** ([render.com](https://render.com)) — New → Blueprint,
   select this repo. Render reads `render.yaml` automatically and prompts
   for `DATABASE_URL` (paste the rewritten Neon string). This config sets
   `CHROMA_EMBEDDED_PATH=/tmp/chroma_data` and
   `RUN_DEMO_SEED_ON_BOOT=true` for you — unlike the Railway path, these
   matter here because Render wipes local disk on every restart/sleep-wake.
3. **Keep it warm** (recommended) — repo Settings → Secrets and variables
   → Actions → Variables, add `RENDER_API_URL` = your Render URL.
   `.github/workflows/keep-api-warm.yml` then pings `/healthz` every 10
   minutes, which matters here specifically because Render *does* sleep
   idle services and Vercel's 10s serverless timeout can collide with
   Render's ~30-60s wake time on a visitor's first request otherwise.
4. **Vercel** — same as step 2 above, `API_BASE_URL` pointed at the
   Render URL instead.

Full step-by-step detail for this path (including exact troubleshooting
for the `/healthz` check) is the same shape as the Railway steps above;
ask if you want it expanded back out here.
