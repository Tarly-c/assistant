from __future__ import annotations

from medical_assistant.graph.state import GraphState
from medical_assistant.schemas import CaseCandidate
from medical_assistant.services.cases.memory import dump_memory, load_memory, remember_question
from medical_assistant.services.cases.planner import select_question, should_finalize


def _candidates_from_state(state: GraphState) -> list[CaseCandidate]:
    return [CaseCandidate(**item) for item in state.get("case_candidates", [])]


def route_after_cases(state: GraphState) -> str:
    memory = load_memory(state.get("case_memory"))
    candidates = _candidates_from_state(state)
    if should_finalize(
        candidates,
        turn_index=state.get("turn_index", 0),
        asked_feature_ids=memory.asked_feature_ids,
    ):
        return "final_answer"
    return "plan_question"


def plan_question_node(state: GraphState) -> dict:
    memory = load_memory(state.get("case_memory"))
    candidates = _candidates_from_state(state)
    question = select_question(
        candidates,
        asked_feature_ids=memory.asked_feature_ids,
        confirmed_feature_ids=list(memory.confirmed_features.keys()),
        denied_feature_ids=list(memory.denied_features.keys()),
    )

    if question is None:
        return {
            "should_answer": True,
            "response_type": "answer",
            "phase": "ANSWERED",
        }

    memory = remember_question(memory, question.model_dump())
    return {
        "response_type": "clarification",
        "answer": question.text,
        "selected_question": question.model_dump(),
        "case_memory": dump_memory(memory),
        "confidence": state.get("confidence", 0.0),
        "phase": "NEEDS_CLARIFICATION",
        "turn_index": state.get("turn_index", 0) + 1,
        "sources": [c.title for c in candidates[:3]],
    }
