"""LangGraph 状态定义。精简到最少字段。"""
from __future__ import annotations

from typing import Any, TypedDict


class S(TypedDict, total=False):
    """Workflow 共享状态。

    字段分三类：
    - 输入：user_input, turn, history
    - 记忆：memory（dict，跨轮持久）
    - 输出：reply, reply_type, probe, matched_case, confidence, candidates
    """

    # ── 输入 ──
    user_input: str              # 用户本轮原始输入
    turn: int                    # 当前轮次
    history: list[dict]          # 对话历史 [{role, content}, ...]

    # ── 跨轮记忆（Memory 的 dict 形式）──
    memory: dict[str, Any]

    # ── 本轮中间/输出 ──
    candidates: list[dict]       # ScoredCase 列表（含评分）
    candidate_count: int
    reply: str                   # 返回给用户的文本
    reply_type: str              # "question" | "answer"
    probe: dict[str, Any]        # 选中的追问详情
    matched_case: dict[str, Any] # 匹配的病例详情
    confidence: float
    best_score: float
    top_candidates: list[dict]   # 给前端的 top N 摘要
