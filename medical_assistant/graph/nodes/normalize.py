"""normalize 节点：轻量翻译/关键词提取，LLM 可选。"""
from __future__ import annotations

from medical_assistant.config import get_settings
from medical_assistant.graph.state import GraphState
from medical_assistant.llm import invoke_structured
from medical_assistant.prompts import NORMALIZE_PROMPT
from medical_assistant.schemas import NormalizedInput
from medical_assistant.services.cases.features import extract_search_terms


def _guess_intent(question: str) -> str:
    q = question or ""
    if any(w in q for w in ("怎么办", "怎么治", "治疗", "处理", "缓解", "吃什么药")):
        return "treatment"
    if any(w in q for w in ("为什么", "原因", "怎么回事", "引起")):
        return "cause"
    if any(w in q for w in ("是不是", "什么病", "诊断", "属于")):
        return "diagnosis"
    if any(w in q for w in ("症状", "表现")):
        return "symptom"
    return "general"


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
    key_terms = []
    for term in (result.key_terms_en or []) + fallback_terms:
        if term and term not in key_terms:
            key_terms.append(term)

    return {
        "query_en": result.query_en or question,
        "intent": result.intent or _guess_intent(question),
        "key_terms_en": key_terms[:10],
    }
