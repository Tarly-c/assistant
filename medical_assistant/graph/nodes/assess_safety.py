from __future__ import annotations

from typing import Any

from medical_assistant.schemas.response import SafetyAssessment
from medical_assistant.schemas.state import ConversationState
from medical_assistant.services.safety.red_flags import assess_red_flags


def assess_safety_node(state: dict[str, Any]) -> dict[str, Any]:
    conversation_state = ConversationState.model_validate(state.get("conversation_state") or {})
    safety = assess_red_flags(conversation_state.medical_context)
    return {"safety": SafetyAssessment.model_validate(safety).model_dump(mode="json")}
