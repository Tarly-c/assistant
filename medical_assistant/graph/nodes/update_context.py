from __future__ import annotations

from typing import Any

from medical_assistant.schemas.input import NormalizedInput
from medical_assistant.schemas.state import ConversationState


def update_context_node(state: dict[str, Any]) -> dict[str, Any]:
    question = state["question"]
    conversation_state = ConversationState.model_validate(state.get("conversation_state") or {})
    normalized = NormalizedInput.model_validate(state.get("normalized") or {})

    conversation_state.register_user_turn(question)
    conversation_state.medical_context.merge_normalized(normalized)
    conversation_state.phase = "NORMALIZED"

    return {"conversation_state": conversation_state.model_dump(mode="json")}
