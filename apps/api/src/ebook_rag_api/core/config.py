from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "ebook-rag-api"
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    cors_allowed_origins: list[str] = ["http://localhost:3000"]
    database_url: str = "sqlite:///./ebook_rag.db"
    uploads_dir: Path = Path("data/uploads")
    max_upload_size_mb: int = 50
    embedding_provider: str = "hashing"
    embedding_dimensions: int = 128
    reranker_provider: str = "token_overlap"
    rerank_candidate_multiplier: int = 4
    answer_provider: str = "extractive"
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "not-needed"
    llm_model: str = "llama3.2"
    llm_timeout_seconds: float = 30.0
    llm_temperature: float = 0.0
    llm_max_tokens: int = 400

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
