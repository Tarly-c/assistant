"""GraphState。"""
from __future__ import annotations
from typing import Any, TypedDict


class S(TypedDict, total=False):
    user_input: str
    turn: int
    history: list[dict]
    memory: dict[str, Any]

    candidates: list[dict]
    candidate_count: int
    reply: str
    reply_type: str              # "question" | "answer"
    probe: dict[str, Any]
    matched_case: dict[str, Any]
    confidence: float
    best_score: float
    top_candidates: list[dict]
