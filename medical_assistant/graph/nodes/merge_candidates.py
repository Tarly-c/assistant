from __future__ import annotations

from typing import Any

from medical_assistant.schemas.retrieval import CandidateTopic, LocalSearchResult
from medical_assistant.schemas.state import ConversationState
from medical_assistant.services.retrieval.ranker import aggregate_candidate_topics


def merge_candidates_node(state: dict[str, Any]) -> dict[str, Any]:
    conversation_state = ConversationState.model_validate(state.get("conversation_state") or {})
    local_result = LocalSearchResult.model_validate(state.get("local_result") or {})

    topics = aggregate_candidate_topics(
        hits=local_result.hits,
        normalized_terms=conversation_state.medical_context.normalized_terms,
        max_topics=3,
    )

    conversation_state.phase = "RETRIEVED"

    return {
        "candidate_topics": [CandidateTopic.model_validate(x).model_dump(mode="json") for x in topics],
        "conversation_state": conversation_state.model_dump(mode="json"),
    }
