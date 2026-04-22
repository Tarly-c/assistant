from __future__ import annotations

from typing import Any

from medical_assistant.schemas.confidence import ConfidenceState
from medical_assistant.schemas.retrieval import LocalSearchResult
from medical_assistant.schemas.state import ConversationState
from medical_assistant.services.retrieval.ranker import topic_gap


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def estimate_confidence_node(state: dict[str, Any]) -> dict[str, Any]:
    conversation_state = ConversationState.model_validate(state.get("conversation_state") or {})
    local_result = LocalSearchResult.model_validate(state.get("local_result") or {})
    candidate_topics = state.get("candidate_topics", []) or []
    safety = state.get("safety", {}) or {}
    medical = conversation_state.medical_context
    bundle = medical.latest_query_bundle

    mapping_confidence = 0.15
    mapping_confidence += 0.35 if medical.latest_query_en else 0.0
    mapping_confidence += 0.05 * min(3, len(bundle.queries) if bundle else 0)
    mapping_confidence += 0.05 * min(4, len(medical.normalized_terms))
    mapping_confidence = _clamp(mapping_confidence)

    top_local_score = local_result.score
    retrieval_confidence = min(1.0, top_local_score / 0.30)
    if candidate_topics:
        retrieval_confidence += 0.04 * min(3, len(candidate_topics[0].get("matched_terms", [])))
    if len(candidate_topics) > 1 and topic_gap(candidate_topics) < 0.03:
        retrieval_confidence -= 0.15
    retrieval_confidence = _clamp(retrieval_confidence)

    unresolved = len(medical.unresolved_questions)
    dialog_confidence = 0.80 if unresolved == 0 else 0.60 if unresolved == 1 else 0.45
    if conversation_state.turn_index >= 2:
        dialog_confidence += 0.05
    dialog_confidence = _clamp(dialog_confidence)

    if safety.get("level") == "high":
        safety_confidence = 0.05
    elif safety.get("missing_checks"):
        safety_confidence = 0.45
    else:
        safety_confidence = 0.90

    overall = (
        0.30 * mapping_confidence
        + 0.35 * retrieval_confidence
        + 0.25 * dialog_confidence
        + 0.10 * safety_confidence
    )
    overall = _clamp(overall)

    reasons: list[str] = []
    if medical.latest_query_en:
        reasons.append("已生成英文检索查询")
    reasons.append(f"查询条数：{len(bundle.queries) if bundle else 0}")
    reasons.append(f"关键词数：{len(medical.normalized_terms)}")
    reasons.append(f"本地最高分：{round(top_local_score, 4)}")
    if len(candidate_topics) > 1:
        reasons.append(f"Top1-Top2 分差：{round(topic_gap(candidate_topics), 4)}")
    if safety.get("missing_checks"):
        reasons.append(f"仍缺安全确认项：{', '.join(safety['missing_checks'])}")

    confidence = ConfidenceState(
        mapping_confidence=round(mapping_confidence, 4),
        retrieval_confidence=round(retrieval_confidence, 4),
        dialog_confidence=round(dialog_confidence, 4),
        safety_confidence=round(safety_confidence, 4),
        overall_confidence=round(overall, 4),
        reasons=reasons,
    )
    conversation_state.confidence = confidence
    return {
        "confidence": confidence.model_dump(mode="json"),
        "conversation_state": conversation_state.model_dump(mode="json"),
    }
