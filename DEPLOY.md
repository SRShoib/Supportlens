# Deploying supportlens for free

**Dashboard: Vercel.** **Backend (Postgres + Chroma + API): Render.**

This repo is already on GitHub (`https://github.com/SRShoib/Supportlens`)
and both Render and Vercel deploy by watching that repo directly — connect
each platform to it via GitHub, no separate upload step.

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

## Why this architecture (Neon + Render, not Render's own Postgres)

- **Render's own free Postgres expires 30 days after creation** (then a
  14-day grace period, then it's deleted). **Neon's free Postgres is
  genuinely permanent** — no expiry, data persists even when idle, only
  compute scales to zero (a brief wake delay, not data loss). So Postgres
  lives on Neon, not Render.
- **Render's free web services have no persistent disk at all** — local
  filesystem changes are wiped on every restart, redeploy, *and*
  spin-down-then-wake. That rules out running Chroma as a normal networked
  service the way `docker-compose.yml` does locally. Instead, this deploy
  target switches `ml/inference/vector_store.py` to an embedded, on-disk
  Chroma client (`CHROMA_EMBEDDED_PATH`) and rebuilds the small index from
  Postgres on every boot (`RUN_DEMO_SEED_ON_BOOT`) — both opt-in env vars,
  both no-ops for local dev/docker-compose, already wired up in
  `render.yaml`.
- **Cold starts stack.** Render's free tier sleeps a service after 15
  minutes of no traffic (~30-60s to wake). Vercel's Hobby plan kills any
  serverless function after 10s. If both have gone idle, the *first* visit
  can fail outright rather than just be slow, because the dashboard's
  server-side fetch to the API dies before Render finishes waking up.
  `.github/workflows/keep-api-warm.yml` (already in this repo) pings
  `/healthz` every 10 minutes to prevent that — set it up in step 3 below.

**Cost:** genuinely $0/month, indefinitely, on both platforms. Trade-offs
for that: Render's free tier sleeps when idle (mitigated by step 3 below),
and it only has 512MB RAM — already the reason transformers aren't served
live on this or any other free/cheap path.

---

## 1. Neon (Postgres)

1. Sign up at [neon.tech](https://neon.tech) — no credit card required.
2. Create a project (any name/region).
3. Copy the connection string from the dashboard — it looks like:
   `postgresql://user:password@ep-xxx.neon.tech/dbname?sslmode=require`
4. **Change the scheme** from `postgresql://` to `postgresql+psycopg://`
   — `apps/api/db/session.py` passes `DATABASE_URL` straight to
   SQLAlchemy, which needs the driver named explicitly (this project uses
   `psycopg` v3, not the default `psycopg2`). Keep the rest of the string
   as-is.
5. Save the rewritten string — you'll paste it into Render next.

## 2. Render (API)

1. Sign up at [render.com](https://render.com) and connect your GitHub
   account.
2. **New > Blueprint**, select the `Supportlens` repo. Render reads
   `render.yaml` from the repo root automatically and prompts for two
   values: `DATABASE_URL` — paste the rewritten Neon connection string
   from step 1 — and `OPENAI_API_KEY` — paste your real OpenAI key here
   (RAG is enabled by default in this config; see the note below on why).
3. Deploy. First build takes a while (installs torch/transformers/
   sentence-transformers/chromadb — the heaviest part of the whole setup,
   expect several minutes).
4. Once live, note the public URL Render assigns, e.g.
   `https://supportlens-api.onrender.com`.
5. Verify: `curl https://<your-render-url>/healthz` should return
   `{"status":"ok","db":"ok",...}`. If it doesn't, check the Render
   service logs — most likely cause is `DATABASE_URL`'s scheme (step 1.4)
   or a Neon connection string that still has a placeholder password.

**About the RAG / suggested-reply feature:** `render.yaml` has
`LLM_ENABLED=true` by default — it goes live as soon as you paste a real
`OPENAI_API_KEY` at deploy time. This is a deliberate choice, not an
oversight: the URL is public once deployed, so anyone who finds it could
run up spend against the app's own `$5` `LLM_BUDGET_USD` guard — the
reason this is safe to leave on is having your own **OpenAI account-level
monthly spending limit** as a hard backstop on the actual dollar exposure
(set one at platform.openai.com if you haven't). The one remaining
trade-off (not a money risk, just availability) is that whichever cap
trips first — yours or the app's — pauses the feature for everyone,
including you, until it resets. If you'd rather avoid that entirely,
leave `OPENAI_API_KEY` blank and manually set `LLM_ENABLED=false` in
Render's dashboard after deploying.

## 3. Keep the API warm (recommended, ~2 minutes)

1. In this GitHub repo: **Settings > Secrets and variables > Actions >
   Variables**.
2. Add a repository variable `RENDER_API_URL` = your Render URL from step
   2.4 (no trailing slash), e.g. `https://supportlens-api.onrender.com`.
3. That's it — `.github/workflows/keep-api-warm.yml` starts pinging
   `/healthz` every 10 minutes automatically, keeping the service inside
   Render's free 750 instance-hours/month (an always-on service uses
   ~730).

## 4. Vercel (dashboard)

1. Sign up at [vercel.com](https://vercel.com), connect GitHub, **Add New
   > Project**, select the `Supportlens` repo.
2. Set **Root Directory** to `apps/dashboard` (this is a monorepo — Vercel
   needs to know where the Next.js app actually lives). It should
   auto-detect the Next.js framework and pnpm (via `package.json`'s
   `packageManager` field) once the root is set correctly.
3. Add an environment variable: `API_BASE_URL` = your Render URL from step
   2.4 (no trailing slash).
4. Deploy.

## 5. Verify end to end

Visit the Vercel URL. Check, in order: **Overview** loads with real stat
tiles, **Tickets** lists real tickets, a **ticket detail** page shows
summary/sentiment/entities, **Topics** shows the volume chart, **Search**
returns results, **Metrics** shows real eval-run numbers, **Try it live**
returns baseline predictions (transformer column will correctly show "not
available" — expected, already-handled degradation, not a bug).

Since RAG is enabled on this deploy: open a ticket and click **"Generate
suggested reply"** — it should return a draft with cited sources within a
few seconds. If it errors instead, the most likely cause is
`OPENAI_API_KEY` not actually being set on the Render service (Environment
tab) despite `LLM_ENABLED=true`.

If the very first load times out: that's the cold-start-stacking issue —
refresh once, or make sure step 3 is actually set up.

## Ongoing cost reality check

- **Vercel Hobby**: $0, indefinitely, for this workload.
- **Neon Free**: $0, indefinitely — 0.5GB storage is far more than the demo
  dataset needs (`data/seed/*.jsonl` is 1.4MB uncompressed).
- **Render Free**: $0 as long as total usage across your Render account
  stays under 750 instance-hours/month combined. One always-on-ish service
  (kept warm by step 3) uses ~730 — leaves ~20 hours of headroom, so avoid
  adding a second always-on free service on the same Render account
  without checking the math.

Nothing here requires a credit card on any of the three platforms as of
this writing — worth double-checking each platform's current pricing page
before you commit, since free-tier terms change without much notice.

---

## Alternative: Railway instead of Render (real cost, better architecture)

This repo also has a tested Railway path (`railway.json`) — a real,
separate, always-on Chroma service with a persistent volume instead of the
embedded-on-boot workaround, and Railway's own Postgres instead of Neon,
since Railway supports persistent volumes and doesn't force services to
sleep. The trade-off: **Railway is not free past a one-time $5/30-day
trial** — after that it's the Hobby plan, $5/month minimum plus metered
usage, likely $5-15/month total for three always-on services. Ask if you
want this path's full steps instead.
