from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the generic case-localization demo."""

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
    debug_llm_payloads: bool = False

    resources_dir: str = "resources"
    chroma_dir: str = "chroma_db"
    trace_dir: str = "traces"
    collection_name: str = "medical_assistant"
    embedding_model: str = "embeddinggemma:latest"
    local_top_k: int = 6
    local_min_score: float = 0.18

    case_data_file: str = "resources/raw/cases_demo.json"
    case_collection_name: str = "case_demo"
    case_initial_top_k: int = 100
    case_display_top_k: int = 5
    case_min_confidence_gap: float = 0.16
    max_clarify_turns: int = 6

    case_question_tree_file: str = "resources/case_question_tree.json"
    tree_max_depth: int = 7
    tree_min_leaf_cases: int = 2
    tree_min_probe_gain: float = 0.08
    tree_probe_options_per_node: int = 3
    tree_use_unknown_as_soft_branch: bool = True
    local_probe_min_gain: float = 0.05

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

    @property
    def case_question_tree_path(self) -> Path:
        return Path(self.case_question_tree_file)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
