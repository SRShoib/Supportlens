# CLAUDE.md — supportlens

Project conventions for Claude Code. Read SPEC.md first — it is the source of truth for scope, module order, and acceptance criteria. This file covers *how* to work, not *what* to build.

## Ground rules

1. **Follow SPEC.md module order (M0 → M10).** Do not start a module before the previous one's acceptance criteria pass. If a criterion seems wrong or infeasible, stop and raise it — don't silently reinterpret it.
2. **Baselines before transformers, always.** Never skip the classical baseline "because the transformer will win anyway." The comparison IS the deliverable.
3. **No training in this repo's runtime.** `ml/training/*` scripts are run directly by the human on their local machine (NVIDIA GPU, VS Code) — never inside Docker or the API process. The API only ever loads exported artifacts from `models/`. Never add GPU deps to the API image.
4. **Paid API discipline (hard rule).**
   - Every OpenAI call goes through `ml/inference/llm_client.py` — a single wrapper with a persisted spend counter and a hard stop at `LLM_BUDGET_USD` (default 5.00).
   - Tests and CI must NEVER hit paid APIs. Mock the wrapper; commit cached fixtures for demo flows.
   - Never introduce a second HTTP path to any LLM provider.
5. **No metric without an eval run.** Any number destined for README/dashboard must be produced by `ml/evaluation/` and persisted to Postgres. Hardcoded metrics are a bug.
6. **Small, reviewable increments.** One module = several small commits (conventional commits: `feat(m3): ...`, `test(m1): ...`), each leaving tests green.

## Environment & tooling

- Python 3.11, `uv` for deps (`uv sync`), single lockfile at repo root for `apps/api` + `ml/`.
- `ml/training` additionally requires a **CUDA-enabled PyTorch build** matching the local NVIDIA driver (installed once per machine, not pinned in the shared lockfile the same way CPU deps are — document the exact install command used in `docs/decisions.md`).
- Lint/format: `ruff check --fix` + `ruff format`. Types: `mypy apps/api ml/inference` (strict-ish; pragmatic ignores allowed in `ml/training`).
- Tests: `pytest` (`make test`). Integration tests use testcontainers for Postgres/Chroma and tiny stub model checkpoints in `tests/fixtures/models/`.
- Frontend: Next.js App Router + Tailwind in `apps/dashboard`, TypeScript strict, `pnpm`.
- Run everything through `make` targets: `make dev`, `make test`, `make up`, `make seed`, `make eval`.

## Code conventions

- Pydantic v2 models in `apps/api/schemas/`; SQLAlchemy 2 models in `apps/api/db/models.py`; Alembic for migrations from M1 onward.
- Inference wrappers in `ml/inference/` expose a uniform interface: `predict(texts: list[str]) -> list[TaskResult]`; model choice (baseline vs transformer) selected by config/request flag, not code duplication.
- Configuration via env vars only (`pydantic-settings`), documented in `.env.example`. No secrets in code or commits.
- Every fine-tuned model gets a model card in `docs/model-cards/` (data, splits, hyperparams, metrics, limitations) written in the same PR that exports the artifact.
- Decision log: any non-obvious design choice gets 3–6 lines in `docs/decisions.md` (date, decision, why, alternatives).

## Data rules

- Raw downloads live outside git (`data/raw/`, gitignored); committed data limited to small fixtures and the 200-example NER gold set.
- All ingestion converges on the canonical schema in SPEC §2. Never let a module read raw source formats directly.
- Random seeds fixed (42) for every split/training run; splits persisted to disk so scores are reproducible.

## Working style with the human

- The builder (Shoib) is expert in FastAPI/Streamlit/Next.js and strong in DSA; explain ML/NLP design trade-offs when they arise, keep web-framework explanations brief.
- **GPU training runs are executed by the human locally in VS Code on their own NVIDIA GPU** — not on Kaggle/Colab. Your job: produce the complete training script + config + a short "how to run this locally" note (env activation command, exact `python ml/training/train_*.py ...` invocation, expected VRAM usage, and rough training time), then integrate the exported artifact when it lands in `models/`.
- When a module is done, output a short acceptance-criteria checklist with evidence (test names, eval run ids) before proposing the next module.
