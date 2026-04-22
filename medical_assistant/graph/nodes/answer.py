from __future__ import annotations

from typing import Any

from medical_assistant.schemas.response import AssistantResponse, SafetyAssessment
from medical_assistant.schemas.retrieval import CandidateTopic, LocalSearchResult, PubMedSearchResult
from medical_assistant.schemas.state import ConversationState
from medical_assistant.services.llm import invoke_text
from medical_assistant.services.questions.templates import (
    build_evidence_items,
    render_candidate_topics_for_user,
    render_evidence_block,
    render_sources_text,
)
from medical_assistant.services.safety.disclaimers import GENERAL_DISCLAIMER, build_risk_text

BASIS_PROMPT = """
你是医学信息整理助手。
只能基于给定证据，简洁总结“为什么这些主题更匹配”和“还不能确定的点”。
不要做诊断，不要给处方，不要编造证据。
输出 2~4 句中文自然语言，不要分点。

用户问题：
{question}

结构化上下文：
{context_json}

候选主题：
{matched_topics}

证据：
{evidence}
""".strip()


def _load_payloads(state: dict[str, Any]):
    conversation_state = ConversationState.model_validate(state.get("conversation_state") or {})
    local_result = LocalSearchResult.model_validate(state.get("local_result") or {})
    web_result = PubMedSearchResult.model_validate(state.get("web_result") or {})
    safety = SafetyAssessment.model_validate(state.get("safety") or {})
    candidate_topics = [
        CandidateTopic.model_validate(x) for x in (state.get("candidate_topics") or [])
    ]
    return conversation_state, local_result, web_result, safety, candidate_topics


def answer_node(state: dict[str, Any]) -> dict[str, Any]:
    conversation_state, local_result, web_result, safety, candidate_topics = _load_payloads(state)

    try:
        basis = invoke_text(
            [
                {"role": "system", "content": "你是医学证据整理助手。"},
                {
                    "role": "user",
                    "content": BASIS_PROMPT.format(
                        question=state["question"],
                        context_json=conversation_state.model_dump_json(indent=2, ensure_ascii=False),
                        matched_topics=render_candidate_topics_for_user(candidate_topics),
                        evidence=render_evidence_block(local_result, web_result),
                    ),
                },
            ]
        ).strip()
    except Exception:
        basis = "当前回答主要基于本地命中的知识条目与检索到的相关证据整理而成，仍有部分细节未完全收敛。"

    sources_text = render_sources_text(local_result, web_result)
    risk_text = build_risk_text(conversation_state.medical_context, safety)
    content = (
        "当前更匹配的知识主题：\n"
        f"{render_candidate_topics_for_user(candidate_topics)}\n\n"
        "依据：\n"
        f"{basis}\n\n"
        "来源：\n"
        f"{sources_text}\n\n"
        "安全提醒：\n"
        f"{risk_text}\n\n"
        f"{GENERAL_DISCLAIMER}"
    )

    conversation_state.phase = "ANSWERED"
    conversation_state.last_response_type = "answer"

    response = AssistantResponse(
        response_type="answer",
        content=content,
        matched_topics=candidate_topics,
        confidence=conversation_state.confidence,
        safety=safety,
        sources=build_evidence_items(local_result, web_result),
    )

    return {
        "conversation_state": conversation_state.model_dump(mode="json"),
        "response": response.model_dump(mode="json"),
    }


def safety_response_node(state: dict[str, Any]) -> dict[str, Any]:
    conversation_state, local_result, web_result, safety, candidate_topics = _load_payloads(state)

    reasons = "；".join(safety.reasons) if safety.reasons else "已出现需要优先排除的危险信号"
    risk_text = build_risk_text(conversation_state.medical_context, safety)
    sources_text = render_sources_text(local_result, web_result)

    content = (
        "当前更需要优先关注安全风险，而不是继续在线细分主题。\n\n"
        f"高风险依据：{reasons}。\n"
        f"建议：{risk_text}\n\n"
        "如正在明显加重，或出现呼吸困难、意识改变、肢体无力、持续胸痛/胸闷等情况，请立即线下就医或求助急救。\n\n"
        "参考来源：\n"
        f"{sources_text}\n\n"
        f"{GENERAL_DISCLAIMER}"
    )

    conversation_state.phase = "SAFETY_ESCALATION"
    conversation_state.last_response_type = "safety"

    response = AssistantResponse(
        response_type="safety",
        content=content,
        matched_topics=candidate_topics,
        confidence=conversation_state.confidence,
        safety=safety,
        sources=build_evidence_items(local_result, web_result),
    )

    return {
        "conversation_state": conversation_state.model_dump(mode="json"),
        "response": response.model_dump(mode="json"),
    }
