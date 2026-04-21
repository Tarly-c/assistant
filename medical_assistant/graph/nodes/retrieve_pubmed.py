from __future__ import annotations

from typing import Any

from medical_assistant.schemas.retrieval import PubMedSearchResult
from medical_assistant.services.retrieval.pubmed import search_pubmed_best


def retrieve_pubmed_node(state: dict[str, Any]) -> dict[str, Any]:
    plan = state.get("plan", {}) or {}
    queries = plan.get("pubmed_queries", []) or []

    if not queries:
        result = PubMedSearchResult(query="", hits=[])
        return {"web_result": result.model_dump(mode="json")}

    web_result = search_pubmed_best(
        question=state["question"],
        queries=queries,
    )
    result = PubMedSearchResult.model_validate(web_result)
    return {"web_result": result.model_dump(mode="json")}
