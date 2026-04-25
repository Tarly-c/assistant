from __future__ import annotations

from medical_assistant.graph.state import GraphState
from medical_assistant.schemas import CaseCandidate
from medical_assistant.services.cases.features import feature_labels
from medical_assistant.services.cases.memory import dump_memory, load_memory
from medical_assistant.services.cases.planner import choose_final_case
from medical_assistant.services.cases.store import confidence_from_candidates


def _candidates_from_state(state: GraphState) -> list[CaseCandidate]:
    return [CaseCandidate(**item) for item in state.get("case_candidates", [])]


def final_answer_node(state: GraphState) -> dict:
    memory = load_memory(state.get("case_memory"))
    candidates = _candidates_from_state(state)
    case = choose_final_case(candidates)

    if case is None:
        answer = "目前没有在本地病例库中找到可匹配的病例。"
        return {
            "response_type": "answer",
            "answer": answer,
            "confidence": 0.0,
            "phase": "ANSWERED",
            "turn_index": state.get("turn_index", 0) + 1,
            "case_memory": dump_memory(memory),
        }

    memory.resolved_case_id = case.case_id
    labels = feature_labels(memory.confirmed_features.keys())
    if not labels:
        labels = feature_labels(case.matched_features or case.feature_tags[:3])

    key_features_text = "、".join(labels[:5]) if labels else "暂无明确确认特征"
    answer = (
        f"更符合本地病例：{case.title}\n"
        f"关键依据：{key_features_text}\n"
        f"处理建议：{case.treat}"
    )

    confidence = confidence_from_candidates(candidates)
    return {
        "response_type": "answer",
        "answer": answer,
        "treatment": case.treat,
        "matched_case": case.model_dump(),
        "resolved_case_id": case.case_id,
        "sources": [case.title],
        "confidence": confidence,
        "phase": "ANSWERED",
        "turn_index": state.get("turn_index", 0) + 1,
        "case_memory": dump_memory(memory),
    }
