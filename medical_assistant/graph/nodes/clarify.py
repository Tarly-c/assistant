from __future__ import annotations

from typing import Any

from medical_assistant.schemas.response import AssistantResponse, SafetyAssessment
from medical_assistant.schemas.retrieval import CandidateTopic
from medical_assistant.schemas.state import ConversationState
from medical_assistant.services.questions.discriminators import choose_clarify_question
from medical_assistant.services.questions.templates import render_candidate_topics_for_user


def clarify_node(state: dict[str, Any]) -> dict[str, Any]:
    conversation_state = ConversationState.model_validate(state.get("conversation_state") or {})
    safety = SafetyAssessment.model_validate(state.get("safety") or {})
    candidate_topics = [
        CandidateTopic.model_validate(x) for x in (state.get("candidate_topics") or [])
    ]

    question = choose_clarify_question(
        conversation_state=conversation_state,
        candidate_topics=candidate_topics,
        safety=safety,
    )

    conversation_state.phase = "NEEDS_CLARIFICATION"
    conversation_state.last_response_type = "clarification"
    conversation_state.last_clarify_question = question
    conversation_state.add_asked_question(question)

    content = (
        "当前更匹配的知识条目：\n"
        f"{render_candidate_topics_for_user(candidate_topics)}\n\n"
        "为了继续缩小范围，我先确认 1 个关键点：\n"
        f"{question}"
    )

    response = AssistantResponse(
        response_type="clarification",
        content=content,
        matched_topics=candidate_topics,
        confidence=conversation_state.confidence,
        safety=safety,
        next_question=question,
    )

    return {
        "conversation_state": conversation_state.model_dump(mode="json"),
        "response": response.model_dump(mode="json"),
    }
