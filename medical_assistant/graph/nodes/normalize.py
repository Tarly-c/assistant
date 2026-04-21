from __future__ import annotations

from typing import Any

from medical_assistant.schemas.input import ConceptCandidate, NormalizedInput
from medical_assistant.schemas.state import ConversationState
from medical_assistant.services.llm import invoke_structured
from medical_assistant.services.terminology.resolver import resolve_terminology


NORMALIZE_PROMPT = """
你是医学输入标准化器。只做信息抽取，不做诊断，不给治疗结论。

请把“最新用户输入”结合“已有结构化上下文”转换成结构化输出。

要求：
1. chief_complaint 用简洁英文医学词，例如 headache / cough / fever / sore throat。
2. normalized_terms 输出 2~6 个英文医学检索词。
3. candidate_concepts 允许多个候选，不要只给一个。
4. facets 的 name 用 snake_case，例如 duration, pain_quality, laterality, nausea, fever。
5. negated_findings 只放用户明确否认的内容。
6. red_flags 只放用户明确提到的高风险信息，不要猜。
7. unresolved_questions 放当前最值得下一轮确认的 1~3 个点。
8. 只输出结构化结果。
""".strip()


def _merge_unique_strs(old: list[str], new: list[str], limit: int | None = None) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in [*(old or []), *(new or [])]:
        text = (item or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
        if limit and len(result) >= limit:
            break
    return result


def _merge_candidates(
    old: list[ConceptCandidate],
    new: list[ConceptCandidate],
) -> list[ConceptCandidate]:
    bucket: dict[str, ConceptCandidate] = {}
    for item in [*(old or []), *(new or [])]:
        key = (item.term or "").strip().lower()
        if not key:
            continue
        previous = bucket.get(key)
        if previous is None or item.support > previous.support:
            bucket[key] = item
    return sorted(bucket.values(), key=lambda x: x.support, reverse=True)[:5]


def _merge_normalized(primary: NormalizedInput, secondary: NormalizedInput) -> NormalizedInput:
    merged = primary.model_copy(deep=True)

    if not merged.original_text:
        merged.original_text = secondary.original_text

    if not merged.chief_complaint:
        merged.chief_complaint = secondary.chief_complaint

    merged.normalized_terms = _merge_unique_strs(
        merged.normalized_terms,
        secondary.normalized_terms,
        limit=8,
    )
    merged.red_flags = _merge_unique_strs(merged.red_flags, secondary.red_flags, limit=8)
    merged.unresolved_questions = _merge_unique_strs(
        merged.unresolved_questions,
        secondary.unresolved_questions,
        limit=4,
    )
    merged.search_keywords = _merge_unique_strs(
        merged.search_keywords,
        secondary.search_keywords,
        limit=8,
    )
    merged.free_text_observations = (
        merged.free_text_observations + secondary.free_text_observations
    )[-5:]

    existing_facets = {f"{f.name}:{f.value}": f for f in merged.facets}
    for item in secondary.facets:
        existing_facets[f"{item.name}:{item.value}"] = item
    merged.facets = list(existing_facets.values())

    existing_negated = {f"{f.name}:{f.value}": f for f in merged.negated_findings}
    for item in secondary.negated_findings:
        existing_negated[f"{item.name}:{item.value}"] = item
    merged.negated_findings = list(existing_negated.values())

    merged.candidate_concepts = _merge_candidates(
        merged.candidate_concepts,
        secondary.candidate_concepts,
    )

    if merged.chief_complaint and not merged.candidate_concepts:
        merged.candidate_concepts = [
            ConceptCandidate(term=merged.chief_complaint, support=0.55, source="llm")
        ]

    return merged


def normalize_node(state: dict[str, Any]) -> dict[str, Any]:
    question = state["question"]
    conversation_state = ConversationState.model_validate(state.get("conversation_state") or {})
    current_turn = conversation_state.turn_index + 1

    try:
        llm_normalized = invoke_structured(
            NormalizedInput,
            [
                {"role": "system", "content": NORMALIZE_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "已有结构化上下文：\n"
                        f"{conversation_state.model_dump_json(indent=2, ensure_ascii=False)}\n\n"
                        "最新用户输入：\n"
                        f"{question}"
                    ),
                },
            ],
        )
    except Exception:
        llm_normalized = NormalizedInput(original_text=question)

    deterministic_normalized = resolve_terminology(
        text=question,
        current_terms=conversation_state.medical_context.normalized_terms,
    )

    normalized = _merge_normalized(llm_normalized, deterministic_normalized)

    if not normalized.original_text:
        normalized.original_text = question

    if normalized.chief_complaint and not normalized.normalized_terms:
        normalized.normalized_terms = [normalized.chief_complaint]

    for facet in normalized.facets:
        facet.turn = current_turn
    for facet in normalized.negated_findings:
        facet.turn = current_turn

    return {"normalized": normalized.model_dump(mode="json")}
