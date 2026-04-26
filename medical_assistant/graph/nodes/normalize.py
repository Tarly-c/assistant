"""normalize node: optional LLM normalization plus generic chunk extraction."""

from __future__ import annotations

from medical_assistant.config import get_settings
from medical_assistant.graph.state import GraphState
from medical_assistant.llm import invoke_structured
from medical_assistant.prompts import NORMALIZE_PROMPT
from medical_assistant.schemas import NormalizedInput
from medical_assistant.services.cases.features import extract_search_terms


def normalize_node(state: GraphState) -> dict:
    question = state.get("question", "")
    settings = get_settings()

    result = NormalizedInput()
    if settings.use_llm_normalize:
        result = invoke_structured(
            NormalizedInput,
            [
                {"role": "system", "content": NORMALIZE_PROMPT},
                {"role": "user", "content": f"当前问题: {question}"},
            ],
        )

    fallback_terms = extract_search_terms(question)
    key_terms: list[str] = []
    for term in (result.key_terms_en or []) + fallback_terms:
        if term and term not in key_terms:
            key_terms.append(term)

    return {
        "query_en": result.query_en or question,
        "intent": result.intent or "general",
        "key_terms_en": key_terms[:10],
    }
