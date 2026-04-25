"""Legacy local retrieval wrapper.

The case demo uses medical_assistant.services.cases.store.search_cases(). This
module is kept for compatibility with old imports.
"""
from __future__ import annotations

from medical_assistant.config import get_settings
from medical_assistant.services.cases.store import search_cases


def search_local(question: str, normalized_terms: list[str] | None = None) -> dict:
    settings = get_settings()
    candidates = search_cases(
        query=question,
        terms=normalized_terms or [],
        top_k=settings.local_top_k,
    )
    hits = [
        {
            "source": c.case_id,
            "title": c.title,
            "snippet": c.description[:700],
            "score": c.score,
            "chunk_id": c.case_id,
        }
        for c in candidates
    ]
    best = candidates[0].score if candidates else 0.0
    return {
        "enough": bool(candidates),
        "score": round(best, 4),
        "reason": "case-level local retrieval",
        "hits": hits,
    }
