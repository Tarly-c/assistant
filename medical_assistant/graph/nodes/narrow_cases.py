from __future__ import annotations

from medical_assistant.graph.state import GraphState
from medical_assistant.services.cases.memory import dump_memory, load_memory
from medical_assistant.services.cases.question_tree import load_question_tree
from medical_assistant.services.cases.store import all_case_ids, apply_feature_filters


def _tree_node_case_ids(node_id: str) -> list[str]:
    if not node_id:
        return []
    tree = load_question_tree()
    if not tree:
        return []
    node = (tree.get("nodes") or {}).get(node_id)
    if not isinstance(node, dict):
        return []
    return list(node.get("case_ids", []) or [])


def narrow_cases_node(state: GraphState) -> dict:
    memory = load_memory(state.get("case_memory"))

    # Prefer the active tree node's case set after a yes/no branch; otherwise use
    # previous candidate set or the full case library.
    tree_ids = _tree_node_case_ids(memory.tree_node_id)
    base_ids = tree_ids or memory.candidate_case_ids or all_case_ids()

    narrowed = apply_feature_filters(
        base_ids,
        confirmed_features=memory.confirmed_features.keys(),
        denied_features=memory.denied_features.keys(),
        probe_splits=memory.probe_splits,
    )
    if not narrowed:
        narrowed = base_ids

    memory.candidate_case_ids = narrowed
    memory.pending_answer_feature = ""
    memory.pending_answer_signal = ""

    return {
        "candidate_case_ids": narrowed,
        "candidate_count": len(narrowed),
        "tree_node_id": memory.tree_node_id,
        "case_memory": dump_memory(memory),
    }
