from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Postgres
    postgres_user: str = "supportlens"
    postgres_password: str = "supportlens"
    postgres_db: str = "supportlens"
    postgres_port: int = 5432
    database_url: str = "postgresql+psycopg://supportlens:supportlens@localhost:5432/supportlens"

    # Chroma
    chroma_host: str = "localhost"
    chroma_port: int = 8001

    # API
    api_port: int = 8000
    env: str = "dev"
    log_level: str = "INFO"

    # Data
    data_dir: str = "./data"
    twcs_csv_path: str = "./data/raw/twcs/twcs.csv"
    hf_home: str = "./data/raw/hf"
    random_seed: int = 42
    slice_target_messages: int = 150_000
    slice_max_brand_share: float = 0.08
    lang_confidence_threshold: float = 0.70

    # LLM (unused until M2; guard defaults to off)
    openai_api_key: str = ""
    llm_budget_usd: float = 5.00
    llm_enabled: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
