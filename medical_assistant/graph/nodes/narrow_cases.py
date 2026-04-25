from __future__ import annotations

from medical_assistant.graph.state import GraphState
from medical_assistant.services.cases.memory import dump_memory, load_memory
from medical_assistant.services.cases.store import all_case_ids, apply_feature_filters


def narrow_cases_node(state: GraphState) -> dict:
    memory = load_memory(state.get("case_memory"))

    base_ids = memory.candidate_case_ids or all_case_ids()
    narrowed = apply_feature_filters(
        base_ids,
        confirmed_features=memory.confirmed_features.keys(),
        denied_features=memory.denied_features.keys(),
    )

    # If the conservative all-feature pass somehow over-narrows, keep the
    # previous base. This should be rare but keeps the demo from dead-ending.
    if not narrowed:
        narrowed = base_ids

    memory.candidate_case_ids = narrowed
    memory.pending_answer_feature = ""
    memory.pending_answer_signal = ""

    return {
        "candidate_case_ids": narrowed,
        "candidate_count": len(narrowed),
        "case_memory": dump_memory(memory),
    }
