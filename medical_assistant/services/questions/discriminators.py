from __future__ import annotations

from medical_assistant.schemas.response import SafetyAssessment
from medical_assistant.schemas.retrieval import CandidateTopic
from medical_assistant.schemas.state import ConversationState


GENERIC_FOLLOW_UPS = {
    "duration": "症状大概持续了多久？是持续存在，还是间歇出现？",
    "severity": "目前最困扰你的表现严重到什么程度？是否已经明显影响活动、进食、睡眠或呼吸？",
    "trigger": "症状出现前后有没有比较明确的诱因或触发因素？",
    "associated": "除了刚才提到的主要不适，还有没有最明显的伴随表现？",
}


def choose_clarify_question(
    conversation_state: ConversationState,
    candidate_topics: list[CandidateTopic],
    safety: SafetyAssessment,
) -> str:
    asked = conversation_state.asked_questions
    medical = conversation_state.medical_context
    bundle = medical.latest_query_bundle
    terms = {item.lower() for item in medical.normalized_terms}

    def not_asked(fragment: str) -> bool:
        return not any(fragment in question for question in asked)

    if safety.missing_checks and not_asked("危险信号"):
        return "为了先排除需要尽快线下处理的情况，能补充一下是否有突然明显加重、意识改变、呼吸困难、持续胸闷胸痛、明显无力或高热不退吗？"

    if bundle is not None:
        categories = {item.category for item in bundle.keywords}
        if "time" not in categories and not_asked("持续了多久"):
            return GENERIC_FOLLOW_UPS["duration"]
        if "severity" not in categories and not_asked("严重到什么程度"):
            return GENERIC_FOLLOW_UPS["severity"]
        if "trigger" not in categories and not_asked("诱因"):
            return GENERIC_FOLLOW_UPS["trigger"]

    if len(candidate_topics) > 1:
        top_titles = " / ".join(item.title for item in candidate_topics[:2])
        if not_asked("伴随表现"):
            return f"当前更接近的方向主要在 {top_titles} 之间。能再补充一下最明显的伴随表现，以及哪些情况会让症状加重或缓解吗？"

    if terms and not_asked("伴随表现"):
        joined = "、".join(list(terms)[:4])
        return f"我已经抓到的核心线索有：{joined}。能再补充一下持续时间、严重程度，以及最明显的伴随表现吗？"

    return "为了继续缩小范围，能再补充一下持续时间、严重程度、诱因，以及最明显的伴随表现吗？"
