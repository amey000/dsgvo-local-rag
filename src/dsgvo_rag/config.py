"""Central configuration. Every value can be overridden via an ``RAG_*`` environment variable."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_PII_ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "IBAN_CODE",
    "CREDIT_CARD",
    "IP_ADDRESS",
    "DE_ADDRESS",
    "DE_TAX_ID",
    "DE_SOCIAL_SECURITY",
]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    # --- Services (only reachable inside the isolated Docker network) ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "dokumente"
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3.1:8b"
    llm_temperature: float = 0.1
    llm_num_ctx: int = 8192
    llm_timeout_s: float = 600.0

    # --- Embeddings ---
    embedding_model: str = "intfloat/multilingual-e5-large"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 16

    # --- PII redaction (Presidio) ---
    spacy_model: str = "de_core_news_md"
    pii_entities: list[str] = Field(default_factory=lambda: list(DEFAULT_PII_ENTITIES))
    pii_score_threshold: float = 0.35
    pii_allow_list: list[str] = Field(default_factory=list)
    redact_queries: bool = True

    # --- Chunking & retrieval ---
    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k: int = 5
    # e5 cosine scores cluster high (~0.7 for unrelated text), so the cut-off sits high too.
    min_score: float = 0.77

    # --- API ---
    max_upload_mb: int = 25


@lru_cache
def get_settings() -> Settings:
    return Settings()
