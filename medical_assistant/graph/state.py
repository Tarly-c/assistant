"""LangGraph 图状态 — 病例定位 demo 版，无 safety / web_search。"""
from __future__ import annotations

from typing import Any, TypedDict


class GraphState(TypedDict, total=False):
    # ── 输入 ──
    question: str
    conversation_history: list[dict[str, Any]]

    # ── normalize 输出 ──
    query_en: str
    intent: str
    key_terms_en: list[str]

    # ── 结构化病例记忆 ──
    case_memory: dict[str, Any]
    confirmed_features: dict[str, Any]
    denied_features: dict[str, Any]
    uncertain_features: dict[str, Any]

    # ── 病例候选集 ──
    candidate_case_ids: list[str]
    case_candidates: list[dict[str, Any]]
    candidate_scores: dict[str, float]
    candidate_count: int
    top_candidates: list[dict[str, Any]]

    # ── 追问计划 ──
    selected_question: dict[str, Any]
    should_answer: bool
    resolved_case_id: str
    matched_case: dict[str, Any]

    # ── 兼容旧输出字段 ──
    hits: list[dict[str, Any]]
    best_score: float
    enough: bool
    sources: list[str]
    confidence: float

    # ── 最终响应 ──
    response_type: str  # answer / clarification
    answer: str
    treatment: str

    # ── 会话 ──
    turn_index: int
    phase: str  # NEW / NEEDS_CLARIFICATION / ANSWERED
