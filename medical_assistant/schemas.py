"""所有数据模型。"""
from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field


# ── LLM 输出 ──

class ConceptItem(BaseModel):
    """单个概念。"""
    term: str = ""
    role: str = ""
    importance: str = "medium"
    negative: bool = False


class NormalizedInput(BaseModel):
    """LLM 从用户输入中抽取的信息。"""
    query_cn: str = ""
    query_en: str = ""
    intent: str = "general"
    concepts: list[ConceptItem] = Field(default_factory=list)


class ExtractedConcepts(BaseModel):
    """LLM 从病例中抽取的概念列表。"""
    concepts: list[ConceptItem] = Field(default_factory=list)


class ParsedAnswer(BaseModel):
    signal: Literal["yes", "no", "uncertain", "unrelated"] = "unrelated"
    confidence: float = 0.0
    evidence: str = ""
    new_observations: list[str] = Field(default_factory=list)


# ── 病例 ──

class CaseRecord(BaseModel):
    case_id: str
    title: str
    description: str
    treat: str
    title_en: str = ""
    description_en: str = ""
    aliases: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    feature_tags: list[str] = Field(default_factory=list)


class ScoredCase(CaseRecord):
    score: float = 0.0
    sentence_sim: float = 0.0      # 句子级语义相似度
    keyword_sim: float = 0.0       # 关键词概念相似度
    probe_score: float = 0.0       # probe 确认/否认得分
    hit_probes: list[str] = Field(default_factory=list)


# ── Probe ──

class Probe(BaseModel):
    probe_id: str
    feature_dim: int = -1          # 对应特征维度（语义簇 0..K-1 或概念维度 K..K+M-1）
    label: str
    text: str                      # LLM 改写后的口语化追问
    positive_ids: list[str] = Field(default_factory=list)
    negative_ids: list[str] = Field(default_factory=list)
    unknown_ids: list[str] = Field(default_factory=list)
    score: float = 0.0
    strategy: str = ""
    tree_node: str = ""
    yes_child: str = ""
    no_child: str = ""
    evidence: list[str] = Field(default_factory=list)
    debug: dict[str, Any] = Field(default_factory=dict)


# ── 会话记忆 ──

class Memory(BaseModel):
    original_input: str = ""
    query_cn: str = ""
    query_en: str = ""
    intent: str = "general"

    # ★ 预计算向量（首轮 embed，跨轮复用）
    query_sentence_vec: list[float] = Field(default_factory=list)
    query_keyword_vecs: list[list[float]] = Field(default_factory=list)  # 用户关键词 embedding 列表

    # Probe 状态
    confirmed: dict[str, dict] = Field(default_factory=dict)
    denied: dict[str, dict] = Field(default_factory=dict)
    uncertain: dict[str, dict] = Field(default_factory=dict)

    candidate_ids: list[str] = Field(default_factory=list)
    asked_probes: list[str] = Field(default_factory=list)

    # 上一轮追问
    last_probe_id: str = ""
    last_probe_text: str = ""
    last_probe_label: str = ""
    last_yes_child: str = ""
    last_no_child: str = ""
    last_tree_node: str = ""

    tree_node: str = ""
    splits: dict[str, dict[str, list[str]]] = Field(default_factory=dict)
    labels: dict[str, str] = Field(default_factory=dict)

    resolved_case_id: str = ""
    turn: int = 0
