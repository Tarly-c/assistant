from __future__ import annotations

from medical_assistant.graph.state import GraphState
from medical_assistant.services.cases.memory import dump_memory, update_from_turn


def update_memory_node(state: GraphState) -> dict:
    memory = update_from_turn(
        state.get("case_memory"),
        question=state.get("question", ""),
        query_en=state.get("query_en", ""),
        key_terms=state.get("key_terms_en", []),
        turn_index=state.get("turn_index", 0),
    )
    return {
        "case_memory": dump_memory(memory),
        "confirmed_features": memory.confirmed_features,
        "denied_features": memory.denied_features,
        "uncertain_features": memory.uncertain_features,
    }
