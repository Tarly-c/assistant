from __future__ import annotations

from typing import Any

from medical_assistant.schemas.retrieval import RetrievalPlan
from medical_assistant.schemas.state import ConversationState
from medical_assistant.services.llm import invoke_structured


PLAN_PROMPT = """
你是医学检索计划器。请根据当前结构化上下文和最新输入，生成检索计划。

要求：
1. local_query_en 必须是英文。
2. normalized_terms 只保留核心医学检索词。
3. pubmed_queries 最多 3 条，偏 guideline / review。
4. intent 只能是 treatment / cause / symptom / diagnosis / general。
5. use_pubmed 默认 false，只有明显需要外部证据时才设 true。
6. 只返回结构化结果。
""".strip()


def plan_node(state: dict[str, Any]) -> dict[str, Any]:
    question = state["question"]
    conversation_state = ConversationState.model_validate(state.get("conversation_state") or {})
    medical = conversation_state.medical_context

    try:
        plan = invoke_structured(
            RetrievalPlan,
            [
                {"role": "system", "content": PLAN_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "当前结构化上下文：\n"
                        f"{conversation_state.model_dump_json(indent=2, ensure_ascii=False)}\n\n"
                        "最新用户输入：\n"
                        f"{question}"
                    ),
                },
            ],
        )
    except Exception:
        plan = RetrievalPlan()

    if not plan.normalized_terms:
        plan.normalized_terms = medical.normalized_terms[:6]

    if not plan.local_query_en:
        if plan.normalized_terms:
            plan.local_query_en = " ".join(plan.normalized_terms[:6])
        elif medical.chief_complaint:
            plan.local_query_en = medical.chief_complaint
        else:
            plan.local_query_en = question

    if not plan.intent:
        plan.intent = "general"

    return {"plan": plan.model_dump(mode="json")}
