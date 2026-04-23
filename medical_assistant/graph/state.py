"""LangGraph 图状态 — 无 safety 字段。"""
from __future__ import annotations
from typing import TypedDict


class GraphState(TypedDict, total=False):
    # ── 输入 ──
    question: str
    conversation_history: list       # [{"role": "user", "content": "..."}, ...]

    # ── normalize 输出 ──
    query_en: str
    intent: str
    key_terms_en: list               # list[str]

    # ── retrieve 输出 ──
    hits: list                       # list[dict]
    best_score: float
    enough: bool

    # ── 最终响应 ──
    response_type: str               # answer / clarification
    answer: str
    sources: list                    # list[str]
    confidence: float

    # ── 会话 ──
    turn_index: int
    phase: str                       # NEW / ANSWERED / NEEDS_CLARIFICATION
