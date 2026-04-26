"""全局配置。所有 LLM 调用始终启用，所有 debug 信息始终打印。"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="MEDICAL_ASSISTANT_", extra="ignore",
    )

    app_name: str = "Medical Assistant"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # LLM
    chat_model: str = "qwen2.5:7b"
    temperature: float = 0.0

    # 病例数据
    case_data_file: str = "resources/raw/cases_demo.json"

    # 向量索引（仅 build_index 使用）
    chroma_dir: str = "chroma_db"
    embedding_model: str = "embeddinggemma:latest"
    case_collection_name: str = "case_demo"

    # 决策树
    tree_file: str = "resources/case_question_tree.json"
    tree_max_depth: int = 7
    tree_min_leaf: int = 2                # 叶节点最少病例数
    tree_min_gain: float = 0.04           # probe 最低信息增益
    tree_probes_per_node: int = 3         # 每个节点保留的 probe 数
    tree_soft_branch: bool = True         # unknown 病例分配到两侧

    # 在线提问
    online_min_gain: float = 0.03         # 在线动态 probe 最低收益
    max_turns: int = 6                    # 最大追问轮数
    min_turns_to_finalize: int = 3        # 至少问几轮才允许终止
    large_set_threshold: int = 10         # 候选集大于此值时不靠分数终止
    confidence_gap: float = 0.16          # top1 - top2 达到此值才允许终止
    display_top_k: int = 5               # 返回给前端的 top 候选数

    @property
    def case_data_path(self) -> Path:
        return Path(self.case_data_file)

    @property
    def tree_path(self) -> Path:
        return Path(self.tree_file)

    @property
    def chroma_path(self) -> Path:
        return Path(self.chroma_dir)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
