from __future__ import annotations

from typing import Any

from medical_assistant.schemas.input import NormalizedInput
from medical_assistant.schemas.retrieval import LocalSearchResult
from medical_assistant.services.retrieval.local import search_local


def retrieve_local_node(state: dict[str, Any]) -> dict[str, Any]:
    normalized = NormalizedInput.model_validate(state.get("normalized") or {}).prepare_for_runtime()

    local_result = search_local(
        question=state["question"],
        queries=normalized.queries,
        normalized_terms=normalized.keyword_terms(limit=12) or normalized.normalized_terms,
    )
    result = LocalSearchResult.model_validate(local_result)
    return {"local_result": result.model_dump(mode="json")}
