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
    chunk_target_words: int = 420
    chunk_min_words: int = 180
    chunk_overlap_words: int = 64
    chunk_max_heading_words: int = 12
    embedding_provider: str = "hashing"
    embedding_dimensions: int = 128
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_base_url: str = "http://localhost:11434/v1"
    embedding_api_key: str = "not-needed"
    embedding_timeout_seconds: float = 30.0
    reranker_provider: str = "token_overlap"
    rerank_candidate_multiplier: int = 4
    retrieval_enable_lexical: bool = True
    retrieval_rrf_k: int = 20
    retrieval_dense_weight: float = 0.6
    retrieval_lexical_weight: float = 0.4
    retrieval_rerank_weight: float = 0.65
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_base_url: str = "http://localhost:11434/v1"
    reranker_api_key: str = "not-needed"
    reranker_timeout_seconds: float = 30.0
    answer_provider: str = "extractive"
    answer_base_url: str | None = None
    answer_api_key: str | None = None
    answer_model: str | None = None
    answer_timeout_seconds: float | None = None
    answer_temperature: float | None = None
    answer_max_tokens: int | None = None
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
