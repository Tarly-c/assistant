from __future__ import annotations

from medical_assistant.schemas.response import SafetyAssessment
from medical_assistant.schemas.state import MedicalContext


def assess_red_flags(context: MedicalContext) -> dict:
    facet_map = {item.name: item for item in context.facets}
    negated_names = {item.name for item in context.negated_findings}
    normalized = {item.lower() for item in context.normalized_terms}
    chief = (context.chief_complaint or "").lower()

    def present(name: str) -> bool:
        item = facet_map.get(name)
        return bool(item and item.status == "present")

    def known(name: str) -> bool:
        return name in facet_map or name in negated_names

    reasons: list[str] = []
    missing_checks: list[str] = []

    if chief == "headache" or "headache" in normalized:
        if present("sudden_onset"):
            reasons.append("突然爆发性头痛")
        elif not known("sudden_onset"):
            missing_checks.append("sudden_onset")

        if present("neck_stiffness") or present("neurologic_deficit"):
            reasons.append("头痛伴颈部发硬或神经系统异常")
        else:
            if not known("neck_stiffness"):
                missing_checks.append("neck_stiffness")
            if not known("neurologic_deficit"):
                missing_checks.append("neurologic_deficit")

    if chief == "cough" or "cough" in normalized:
        if present("shortness_of_breath") or present("chest_pain"):
            reasons.append("咳嗽伴气短或胸痛")
        else:
            if not known("shortness_of_breath"):
                missing_checks.append("shortness_of_breath")
            if not known("chest_pain"):
                missing_checks.append("chest_pain")

    if chief == "fever" or "fever" in normalized:
        if present("neck_stiffness") or present("neurologic_deficit"):
            reasons.append("发热伴颈部发硬或意识/神经异常")
        else:
            if not known("neck_stiffness"):
                missing_checks.append("neck_stiffness")
            if not known("neurologic_deficit"):
                missing_checks.append("neurologic_deficit")

    if context.red_flags:
        for item in context.red_flags:
            if item not in reasons:
                reasons.append(item)

    assessment = SafetyAssessment(
        level="high" if reasons else "normal",
        reasons=reasons[:5],
        missing_checks=missing_checks[:3],
    )
    return assessment.model_dump(mode="json")
