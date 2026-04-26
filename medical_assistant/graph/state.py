"""LangGraph state。"""
from __future__ import annotations

from typing import Any, TypedDict


class GraphState(TypedDict, total=False):
    question: str
    conversation_history: list[dict[str, Any]]
    query_en: str
    intent: str
    key_terms_en: list[str]
    case_memory: dict[str, Any]
    confirmed_features: dict[str, Any]
    denied_features: dict[str, Any]
    uncertain_features: dict[str, Any]
    tree_node_id: str
    candidate_case_ids: list[str]
    case_candidates: list[dict[str, Any]]
    candidate_scores: dict[str, float]
    candidate_count: int
    top_candidates: list[dict[str, Any]]
    selected_question: dict[str, Any]
    should_answer: bool
    resolved_case_id: str
    matched_case: dict[str, Any]
    hits: list[dict[str, Any]]
    best_score: float
    enough: bool
    sources: list[str]
    confidence: float
    response_type: str
    answer: str
    treatment: str
    turn_index: int
    phase: str
