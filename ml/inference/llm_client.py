import hashlib
from dataclasses import dataclass

from api.config import Settings, get_settings
from api.db.models import LLMCall
from openai import OpenAI
from sqlalchemy import func, select
from sqlalchemy.orm import Session

# Per-1M-token pricing for the one small model every paid call in this repo
# uses (CLAUDE.md: "never introduce a second HTTP path to any LLM provider").
# Update here if the model changes — nowhere else.
MODEL = "gpt-4o-mini"
_PRICE_PER_1M_INPUT_TOKENS = 0.15
_PRICE_PER_1M_OUTPUT_TOKENS = 0.60


class LLMDisabledError(RuntimeError):
    pass


class BudgetExceededError(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMCallResult:
    response: str
    cached: bool
    cost_usd: float


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def estimate_cost_usd(prompt_tokens: int, completion_tokens: int) -> float:
    return (
        prompt_tokens / 1_000_000 * _PRICE_PER_1M_INPUT_TOKENS
        + completion_tokens / 1_000_000 * _PRICE_PER_1M_OUTPUT_TOKENS
    )


def is_over_budget(current_spend_usd: float, budget_usd: float) -> bool:
    return current_spend_usd >= budget_usd


class LLMClient:
    """The only HTTP path to any LLM provider in this repo (CLAUDE.md hard
    rule). Every call is cached by (purpose, model, prompt_hash) in
    `llm_calls` — a repeated prompt reuses the cached response instead of
    re-billing — and the persisted spend counter (SUM(cost_usd)) is checked
    before every *new* call, hard-stopping at `settings.llm_budget_usd`.
    Refuses to do anything at all while `LLM_ENABLED` is false, which is the
    default until a human explicitly opts in."""

    def __init__(
        self,
        session: Session,
        settings: Settings | None = None,
        openai_client: OpenAI | None = None,
    ) -> None:
        self._session = session
        self._settings = settings or get_settings()
        self._openai_client = openai_client

    def _client(self) -> OpenAI:
        if self._openai_client is None:
            self._openai_client = OpenAI(api_key=self._settings.openai_api_key)
        return self._openai_client

    def total_spend_usd(self) -> float:
        stmt = select(func.coalesce(func.sum(LLMCall.cost_usd), 0.0))
        return float(self._session.execute(stmt).scalar_one())

    def _cached(self, purpose: str, model: str, hashed: str) -> LLMCall | None:
        stmt = select(LLMCall).where(
            LLMCall.purpose == purpose, LLMCall.model == model, LLMCall.prompt_hash == hashed
        )
        return self._session.scalars(stmt).first()

    def complete(self, *, purpose: str, prompt: str, system: str | None = None) -> LLMCallResult:
        if not self._settings.llm_enabled:
            raise LLMDisabledError(
                "LLM_ENABLED is false; refusing to call a paid API. Set it in .env to proceed."
            )

        hashed = prompt_hash(prompt)
        cached = self._cached(purpose, MODEL, hashed)
        if cached is not None:
            return LLMCallResult(response=cached.response, cached=True, cost_usd=0.0)

        current_spend = self.total_spend_usd()
        if is_over_budget(current_spend, self._settings.llm_budget_usd):
            raise BudgetExceededError(
                f"LLM_BUDGET_USD={self._settings.llm_budget_usd} reached "
                f"(spent ${current_spend:.4f}); refusing further calls."
            )

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self._client().chat.completions.create(
            model=MODEL,
            messages=messages,  # type: ignore[arg-type]
            temperature=0,
        )
        text = response.choices[0].message.content or ""
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        cost = estimate_cost_usd(prompt_tokens, completion_tokens)

        self._session.add(
            LLMCall(
                purpose=purpose,
                model=MODEL,
                prompt_hash=hashed,
                response=text,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=cost,
            )
        )
        self._session.commit()

        return LLMCallResult(response=text, cached=False, cost_usd=cost)
