from __future__ import annotations

from typing import Any

from medical_assistant.schemas.retrieval import LocalSearchResult
from medical_assistant.services.retrieval.local import search_local


def retrieve_local_node(state: dict[str, Any]) -> dict[str, Any]:
    plan = state.get("plan", {}) or {}
    local_result = search_local(
        question=state["question"],
        local_query=plan.get("local_query_en") or state["question"],
        normalized_terms=plan.get("normalized_terms", []) or [],
    )
    result = LocalSearchResult.model_validate(local_result)
    return {"local_result": result.model_dump(mode="json")}
