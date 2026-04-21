from __future__ import annotations

from medical_assistant.services.retrieval.ranker import topic_gap


def route_after_confidence(state: dict) -> str:
    safety = state.get("safety", {}) or {}
    confidence = state.get("confidence", {}) or {}
    local_result = state.get("local_result", {}) or {}
    plan = state.get("plan", {}) or {}
    candidate_topics = state.get("candidate_topics", []) or []

    overall = float(confidence.get("overall_confidence", 0.0))

    if safety.get("level") == "high":
        return "safety"

    if overall < 0.58:
        return "clarify"

    if safety.get("missing_checks") and overall < 0.82:
        return "clarify"

    if not local_result.get("enough", False):
        return "search_pubmed" if plan.get("use_pubmed", False) else "clarify"

    if len(candidate_topics) > 1 and topic_gap(candidate_topics) < 0.03:
        return "clarify"

    return "answer"
