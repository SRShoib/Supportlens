from unittest.mock import MagicMock

import pytest
from api.config import Settings

from ml.inference.llm_client import (
    LLMClient,
    LLMDisabledError,
    estimate_cost_usd,
    is_over_budget,
    prompt_hash,
)


def test_prompt_hash_deterministic() -> None:
    assert prompt_hash("classify this") == prompt_hash("classify this")


def test_prompt_hash_differs_for_different_prompts() -> None:
    assert prompt_hash("a") != prompt_hash("b")


def test_estimate_cost_zero_tokens() -> None:
    assert estimate_cost_usd(0, 0) == 0.0


def test_estimate_cost_scales_with_tokens() -> None:
    assert estimate_cost_usd(2_000_000, 0) > estimate_cost_usd(1_000_000, 0)


def test_estimate_cost_output_priced_higher_than_input() -> None:
    # completion tokens cost 4x input tokens for gpt-4o-mini pricing
    assert estimate_cost_usd(0, 1_000_000) > estimate_cost_usd(1_000_000, 0)


def test_is_over_budget_below_cap() -> None:
    assert is_over_budget(0.10, 0.50) is False


def test_is_over_budget_at_cap() -> None:
    assert is_over_budget(0.50, 0.50) is True


def test_is_over_budget_above_cap() -> None:
    assert is_over_budget(1.00, 0.50) is True


def test_complete_raises_when_llm_disabled() -> None:
    settings = Settings(_env_file=None, llm_enabled=False)  # type: ignore[call-arg]
    client = LLMClient(session=MagicMock(), settings=settings, openai_client=MagicMock())

    with pytest.raises(LLMDisabledError):
        client.complete(purpose="urgency_llm_seed", prompt="hello")


def test_complete_never_touches_openai_client_when_disabled() -> None:
    settings = Settings(_env_file=None, llm_enabled=False)  # type: ignore[call-arg]
    openai_client = MagicMock()
    client = LLMClient(session=MagicMock(), settings=settings, openai_client=openai_client)

    with pytest.raises(LLMDisabledError):
        client.complete(purpose="urgency_llm_seed", prompt="hello")

    openai_client.chat.completions.create.assert_not_called()
