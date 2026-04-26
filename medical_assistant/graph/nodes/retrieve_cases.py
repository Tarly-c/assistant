from __future__ import annotations

from medical_assistant.graph.state import GraphState
from medical_assistant.services.cases.memory import dump_memory, load_memory
from medical_assistant.services.cases.store import (
    confidence_from_candidates,
    search_cases,
    top_candidate_summary,
)


def retrieve_cases_node(state: GraphState) -> dict:
    memory = load_memory(state.get("case_memory"))
    query = state.get("query_en") or state.get("question", "")
    terms = state.get("key_terms_en", []) or memory.key_terms

    candidates = search_cases(
        query=query,
        terms=terms,
        candidate_ids=state.get("candidate_case_ids") or memory.candidate_case_ids or None,
        confirmed_features=memory.confirmed_features.keys(),
        denied_features=memory.denied_features.keys(),
        probe_splits=memory.probe_splits,
        top_k=0,  # keep the whole current feasible set for question planning
    )

    memory.candidate_case_ids = [case.case_id for case in candidates]
    best_score = candidates[0].score if candidates else 0.0
    confidence = confidence_from_candidates(candidates)
    top = top_candidate_summary(candidates)

    hits = [
        {
            "case_id": case.case_id,
            "title": case.title,
            "snippet": case.description[:240],
            "score": case.score,
        }
        for case in candidates[:5]
    ]

    return {
        "case_candidates": [case.model_dump() for case in candidates],
        "candidate_case_ids": memory.candidate_case_ids,
        "candidate_scores": {case.case_id: case.score for case in candidates},
        "candidate_count": len(candidates),
        "top_candidates": top,
        "hits": hits,
        "best_score": best_score,
        "confidence": confidence,
        "enough": bool(candidates),
        "case_memory": dump_memory(memory),
    }
