"""retrieve 节点：规则驱动，不调用 LLM。"""
from __future__ import annotations

from medical_assistant.graph.state import GraphState
from medical_assistant.services.retrieval.local import search_local
from medical_assistant.config import get_settings


def retrieve_node(state: GraphState) -> dict:
    query_en = state.get("query_en", state.get("question", ""))
    key_terms = state.get("key_terms_en", [])
    settings = get_settings()

    raw = search_local(
        question=query_en,
        normalized_terms=key_terms,
    )
    # raw 是 dict: {enough, score, reason, hits: [...]}

    hits = raw.get("hits", [])
    best_score = raw.get("score", 0.0)

    return {
        "hits": hits,
        "best_score": best_score,
        "enough": best_score >= settings.local_min_score,
    }
