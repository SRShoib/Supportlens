from unittest.mock import MagicMock

import pytest
from api.config import Settings
from sqlalchemy.orm import Session

from ml.inference.llm_client import BudgetExceededError, LLMClient

pytestmark = pytest.mark.integration


def _mock_response(text: str, prompt_tokens: int = 10, completion_tokens: int = 2) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=text))]
    response.usage = MagicMock(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    return response


def _settings(**overrides: object) -> Settings:
    defaults = {"llm_enabled": True, "llm_budget_usd": 5.0, "openai_api_key": "test-key"}
    return Settings(_env_file=None, **{**defaults, **overrides})  # type: ignore[call-arg,arg-type]


def test_repeated_prompt_reuses_cache_without_a_new_openai_call(db_session: Session) -> None:
    openai_client = MagicMock()
    openai_client.chat.completions.create.return_value = _mock_response("high")
    client = LLMClient(db_session, _settings(), openai_client)

    first = client.complete(purpose="urgency_llm_seed", prompt="urgent message")
    second = client.complete(purpose="urgency_llm_seed", prompt="urgent message")

    assert first.cached is False
    assert second.cached is True
    assert second.response == "high"
    assert openai_client.chat.completions.create.call_count == 1


def test_spend_accumulates_across_distinct_prompts(db_session: Session) -> None:
    openai_client = MagicMock()
    openai_client.chat.completions.create.return_value = _mock_response(
        "low", prompt_tokens=1000, completion_tokens=1000
    )
    client = LLMClient(db_session, _settings(), openai_client)

    client.complete(purpose="urgency_llm_seed", prompt="message one")
    client.complete(purpose="urgency_llm_seed", prompt="message two")

    assert client.total_spend_usd() > 0
    assert openai_client.chat.completions.create.call_count == 2


def test_max_tokens_is_forwarded_to_the_openai_call(db_session: Session) -> None:
    openai_client = MagicMock()
    openai_client.chat.completions.create.return_value = _mock_response("short reply")
    client = LLMClient(db_session, _settings(), openai_client)

    client.complete(purpose="rag_reply_draft", prompt="draft a reply", max_tokens=400)

    _, kwargs = openai_client.chat.completions.create.call_args
    assert kwargs["max_tokens"] == 400


def test_max_tokens_defaults_to_none_when_not_passed(db_session: Session) -> None:
    openai_client = MagicMock()
    openai_client.chat.completions.create.return_value = _mock_response("short reply")
    client = LLMClient(db_session, _settings(), openai_client)

    client.complete(purpose="urgency_llm_seed", prompt="classify this")

    _, kwargs = openai_client.chat.completions.create.call_args
    assert kwargs["max_tokens"] is None


def test_budget_exceeded_raises_and_stops_further_calls(db_session: Session) -> None:
    openai_client = MagicMock()
    openai_client.chat.completions.create.return_value = _mock_response(
        "low", prompt_tokens=1000, completion_tokens=1000
    )
    client = LLMClient(db_session, _settings(llm_budget_usd=0.000001), openai_client)

    client.complete(purpose="urgency_llm_seed", prompt="first message")

    with pytest.raises(BudgetExceededError):
        client.complete(purpose="urgency_llm_seed", prompt="second message")

    assert openai_client.chat.completions.create.call_count == 1
