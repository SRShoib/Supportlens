# supportlens dashboard

Next.js (App Router) + Tailwind analytics UI for supportlens. See the repo root's `SPEC.md` and
`CLAUDE.md` for project-wide context and conventions.

## Scope (M3.5)

A deliberately small scaffold: a ticket list (`/tickets`) and ticket detail view (`/tickets/[id]`)
reading the existing `GET /tickets` API, built so later modules (M4's entity chips, M5's sentiment
sparkline, M7's topic charts, M9's metrics dashboard) have a real surface to extend rather than
starting from scratch each time.

## Running locally

```bash
cp .env.example .env.local   # API_BASE_URL defaults to http://localhost:8000
pnpm install
pnpm dev
```

Requires `apps/api` running separately (`make dev` from the repo root) and reachable at
`API_BASE_URL`.

## Running via docker compose

`make up` from the repo root brings this up alongside `api`, `postgres`, and `chroma` — see
`infra/docker-compose.yml` and `infra/dashboard.Dockerfile`.

## Conventions

TypeScript strict, Tailwind for styling, Server Components fetch `apps/api` directly
(`src/lib/api.ts`) — no client-side API calls, so there's no CORS configuration needed on the
FastAPI side. `pnpm lint` / `pnpm build` before committing.
