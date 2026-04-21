from __future__ import annotations

from medical_assistant.schemas.response import SafetyAssessment
from medical_assistant.schemas.state import MedicalContext

GENERAL_DISCLAIMER = "以上内容仅作健康信息参考，不替代线下面诊、体格检查、检验或正式诊断。"


def build_risk_text(context: MedicalContext, safety: SafetyAssessment) -> str:
    if safety.level == "high":
        return "当前描述里已经出现高风险信号，建议尽快线下就医；若出现意识改变、肢体无力、呼吸困难或持续恶化，请立即求助急救。"

    chief = (context.chief_complaint or "").lower()

    if chief == "headache":
        return "若头痛突然达到最剧烈、伴发热和颈部发硬、说话不清、肢体无力、抽搐或明显意识改变，应尽快就医。"
    if chief == "cough":
        return "若咳嗽伴明显气短、胸痛、发绀、持续高热，或症状迅速加重，应尽快就医。"
    if chief == "fever":
        return "若高热持续不退、伴意识改变、颈部发硬、明显脱水或呼吸困难，应尽快就医。"

    return "若症状明显加重、持续时间过长，或出现呼吸困难、意识改变、肢体无力等情况，应尽快就医。"
