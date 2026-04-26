"""所有 Pydantic 数据模型。"""
from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


# ── LLM 输出模型 ──

class NormalizedInput(BaseModel):
    """normalize 节点让 LLM 返回的结构。"""
    query_en: str = ""                    # 英文检索查询
    intent: str = "general"               # 意图分类
    key_terms: list[str] = Field(default_factory=list)  # 关键术语


class ParsedAnswer(BaseModel):
    """LLM 对用户回答的解析结果。"""
    signal: Literal["yes", "no", "uncertain", "unrelated"] = "unrelated"
    confidence: float = 0.0
    evidence: str = ""                    # 判断理由
    new_observations: list[str] = Field(default_factory=list)  # 用户额外提供的症状


# ── 病例模型 ──

class CaseRecord(BaseModel):
    """一条病例。"""
    case_id: str
    title: str                            # 病例名称，如"可复性牙髓炎"
    description: str                      # 症状描述
    treat: str                            # 处理建议
    # 可选多语言字段
    title_en: str = ""
    description_en: str = ""
    aliases: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    key_terms_en: list[str] = Field(default_factory=list)
    search_terms: list[str] = Field(default_factory=list)
    feature_tags: list[str] = Field(default_factory=list)


class ScoredCase(CaseRecord):
    """带评分的候选病例。"""
    score: float = 0.0
    hit_probes: list[str] = Field(default_factory=list)   # 命中的 probe IDs
    hit_terms: list[str] = Field(default_factory=list)     # 命中的搜索词


# ── 提问模型 ──

class Probe(BaseModel):
    """一个追问（离线树 probe 或在线动态 probe）。"""
    probe_id: str                         # 唯一标识
    label: str                            # 可读标签（≤36 字）
    text: str                             # 追问文本
    positive_ids: list[str] = Field(default_factory=list)   # "是" → 保留的病例
    negative_ids: list[str] = Field(default_factory=list)   # "否" → 保留的病例
    unknown_ids: list[str] = Field(default_factory=list)    # 不确定的病例
    score: float = 0.0                    # 切分质量分
    strategy: str = ""                    # tree_probe / online_probe / confirm
    tree_node: str = ""                   # 所在树节点
    yes_child: str = ""                   # yes 分支的子节点
    no_child: str = ""                    # no 分支的子节点
    evidence: list[str] = Field(default_factory=list)       # 支撑证据文本
    debug: dict[str, Any] = Field(default_factory=dict)


# ── 会话记忆 ──

class Memory(BaseModel):
    """跨轮次结构化记忆。唯一的持久状态。"""

    # 原始信息
    original_input: str = ""              # 用户首次输入（如"我牙疼"）
    search_query: str = ""                # 持久搜索查询（不会被"是的"覆盖）
    search_terms: list[str] = Field(default_factory=list)
    intent: str = "general"

    # Probe 确认状态
    confirmed: dict[str, dict] = Field(default_factory=dict)   # probe_id → {evidence}
    denied: dict[str, dict] = Field(default_factory=dict)
    uncertain: dict[str, dict] = Field(default_factory=dict)

    # 候选集
    candidate_ids: list[str] = Field(default_factory=list)

    # 已问过的 probe（防重复）
    asked_probes: list[str] = Field(default_factory=list)

    # 上一轮追问（用于解析本轮回答）
    last_probe_id: str = ""
    last_probe_text: str = ""
    last_probe_label: str = ""
    last_yes_child: str = ""
    last_no_child: str = ""
    last_tree_node: str = ""

    # 决策树位置
    tree_node: str = ""

    # Probe 切分数据（每个 probe 把病例分成 pos/neg/unk 三组）
    splits: dict[str, dict[str, list[str]]] = Field(default_factory=dict)
    labels: dict[str, str] = Field(default_factory=dict)

    # 结果
    resolved_case_id: str = ""
    turn: int = 0
