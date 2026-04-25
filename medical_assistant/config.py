from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings.

    The demo keeps the original app/LLM/Chroma settings, then adds a small set
    of generic case-localization settings. Nothing here is toothache-specific;
    the current JSON file is only the first demo dataset.
    """

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
    use_llm_normalize: bool = False

    chroma_dir: str = "chroma_db"
    trace_dir: str = "traces"
    collection_name: str = "medical_assistant"
    embedding_model: str = "embeddinggemma:latest"
    local_top_k: int = 6
    local_min_score: float = 0.18

    # Generic case-demo settings.
    case_data_file: str = "resources/raw/cases_demo.json"
    case_collection_name: str = "case_demo"
    case_initial_top_k: int = 100
    case_display_top_k: int = 5
    case_min_confidence_gap: float = 0.16
    max_clarify_turns: int = 6

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
    def case_data_path(self) -> Path:
        return Path(self.case_data_file)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
