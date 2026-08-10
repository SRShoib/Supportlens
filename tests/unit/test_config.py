from pathlib import Path

from api.config import Settings

ENV_EXAMPLE = Path(__file__).resolve().parents[2] / ".env.example"


def test_settings_parses_env_example() -> None:
    settings = Settings(_env_file=ENV_EXAMPLE)  # type: ignore[call-arg]

    assert settings.postgres_user == "supportlens"
    assert settings.postgres_db == "supportlens"
    assert settings.random_seed == 42
    assert settings.llm_enabled is False
    assert settings.openai_api_key == ""


def test_database_url_contains_configured_port() -> None:
    settings = Settings(_env_file=ENV_EXAMPLE)  # type: ignore[call-arg]

    assert settings.database_url.startswith("postgresql+psycopg://")
    assert f":{settings.postgres_port}/" in settings.database_url


def test_defaults_used_when_no_env_file() -> None:
    settings = Settings(_env_file="/nonexistent/path/.env")  # type: ignore[call-arg]

    assert settings.postgres_user == "supportlens"
    assert settings.env == "dev"
