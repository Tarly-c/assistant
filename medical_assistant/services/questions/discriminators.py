from __future__ import annotations

from medical_assistant.schemas.response import SafetyAssessment
from medical_assistant.schemas.retrieval import CandidateTopic
from medical_assistant.schemas.state import ConversationState


def choose_clarify_question(
    conversation_state: ConversationState,
    candidate_topics: list[CandidateTopic],
    safety: SafetyAssessment,
) -> str:
    medical = conversation_state.medical_context
    chief = (medical.chief_complaint or "").lower()
    facet_names = medical.facet_names()
    asked = conversation_state.asked_questions
    topic_titles = " ".join(item.title.lower() for item in candidate_topics)
    normalized_terms = {item.lower() for item in medical.normalized_terms}

    def not_asked(fragment: str) -> bool:
        return not any(fragment in q for q in asked)

    if chief == "headache" or "headache" in normalized_terms or "migraine" in topic_titles:
        if "sudden_onset" not in facet_names and not_asked("突然一下达到最严重"):
            return "为了先排除需要尽快就医的情况：这次头痛是突然一下达到最严重，还是逐渐加重的？"
        if "fever" not in facet_names and "neck_stiffness" not in facet_names and not_asked("发热、颈部发硬"):
            return "是否伴有发热、颈部发硬，或者说话含糊、肢体无力？"
        if "pain_quality" not in facet_names and not_asked("搏动痛"):
            return "你的头痛更像一跳一跳的搏动痛，还是像紧箍一样的压迫痛？"
        if "laterality" not in facet_names and not_asked("一侧"):
            return "疼痛主要在一侧，还是两侧/整个头部？"

    if chief == "cough" or "cough" in normalized_terms:
        if "shortness_of_breath" not in facet_names and not_asked("气短"):
            return "咳嗽时是否伴有气短、胸痛，或者明显呼吸费力？"
        if "duration" not in facet_names and not_asked("持续多久"):
            return "咳嗽已经持续多久了？是干咳还是有痰？"

    if chief == "fever" or "fever" in normalized_terms:
        if "temperature" not in facet_names and not_asked("最高体温"):
            return "体温最高大概到多少度？已经持续多久了？"
        if "rash" not in facet_names and not_asked("皮疹"):
            return "除了发热，还伴有皮疹、咽痛、咳嗽或腹泻吗？"

    if safety.missing_checks:
        return "为了先排除需要尽快就医的情况，能补充一下是否有突然加重、意识改变、肢体无力、呼吸困难或持续高热吗？"

    return "能再补充一下持续时间、严重程度，以及最明显的伴随症状吗？"
