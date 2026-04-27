"""全局配置。"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="MA_", extra="ignore",
    )

    app_name: str = "Medical Assistant"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # LLM
    chat_model: str = "qwen2.5:7b"
    temperature: float = 0.0

    # Embedding（多语言模型，中英文通用）
    embedding_model: str = "bge-m3:latest"
    embedding_batch: int = 64

    # 数据路径
    case_file: str = "resources/raw/cases_demo.json"
    vectors_file: str = "resources/case_vectors.json"
    clusters_file: str = "resources/feature_clusters.json"
    tree_file: str = "resources/case_question_tree.json"

    # 离线特征空间
    max_semantic_clusters: int = 200   # 句子级语义簇上限
    max_concept_dims: int = 80         # 关键词概念维度上限
    semantic_cluster_th: float = 0.78  # 语义簇聚类阈值
    concept_cluster_th: float = 0.75   # 概念维度聚类阈值

    # 决策树
    tree_max_depth: int = 7
    tree_min_leaf: int = 2
    tree_min_gain: float = 0.04
    tree_probes_per_node: int = 3
    tree_soft_branch: bool = True

    # 在线提问
    online_min_gain: float = 0.03
    max_turns: int = 6
    min_turns_to_finalize: int = 3
    large_set_threshold: int = 10
    confidence_gap: float = 0.16
    display_top_k: int = 5

    @property
    def case_path(self) -> Path:
        return Path(self.case_file)

    @property
    def vectors_path(self) -> Path:
        return Path(self.vectors_file)

    @property
    def clusters_path(self) -> Path:
        return Path(self.clusters_file)

    @property
    def tree_path(self) -> Path:
        return Path(self.tree_file)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
