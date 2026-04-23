from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="MEDICAL_ASSISTANT_",
        extra="ignore",
    )

    app_name: str = "Medical Assistant"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    chat_model: str = "qwen2.5:7b"
    temperature: float = 0.0

    resources_dir: str = "resources/medlineplus/retrieval_units"
    chroma_dir: str = "chroma_db"
    trace_dir: str = "traces"

    collection_name: str = "medical_assistant"
    embedding_model: str = "embeddinggemma:latest"


    local_top_k: int = 6
    local_min_score: float = 0.18

    pubmed_retmax: int = 8
    pubmed_min_score: float = 2.0
    pubmed_tool_name: str = "medical-assistant"
    pubmed_email: str = ""
    pubmed_api_key: str = ""

    mesh_cache_file: str = "resources/medlineplus/mesh_terms.jsonl"

    @property
    def resources_path(self) -> Path:
        return Path(self.resources_dir)

    @property
    def chroma_path(self) -> Path:
        return Path(self.chroma_dir)

    @property
    def trace_path(self) -> Path:
        return Path(self.trace_dir)

    @property
    def mesh_cache_path(self) -> Path:
        return Path(self.mesh_cache_file)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
