"""LLM 回答分类：判断用户回答是 yes / no / uncertain / unrelated。"""
from __future__ import annotations

import json
import os
from typing import Any

from medical_assistant.llm import invoke_structured
from medical_assistant.schemas import ProbeAnswerParse

SYSTEM_PROMPT = """
你是问诊回答解析器。判断用户回答是否确认了上一轮追问。
只输出 signal/confidence/evidence/new_observations。
signal: yes / no / uncertain / unrelated
""".strip()


def build_payload(*, question_text: str, user_answer: str,
                  probe_label: str = "", evidence_texts: list[str] | None = None,
                  ) -> list[dict[str, str]]:
    data: dict[str, Any] = {
        "previous_question": question_text,
        "probe_label": probe_label,
        "probe_evidence_texts": evidence_texts or [],
        "user_answer": user_answer,
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(data, ensure_ascii=False, indent=2)},
    ]


def parse_probe_answer(*, question_text: str, user_answer: str,
                       probe_label: str = "", evidence_texts: list[str] | None = None,
                       ) -> ProbeAnswerParse:
    msgs = build_payload(question_text=question_text, user_answer=user_answer,
                         probe_label=probe_label, evidence_texts=evidence_texts)
    if os.getenv("MEDICAL_ASSISTANT_DEBUG_LLM_PAYLOADS", "").lower() in {"1", "true", "yes"}:
        print("[LLM payload] answer_parser")
        print(json.dumps(msgs, ensure_ascii=False, indent=2))
    result = invoke_structured(ProbeAnswerParse, msgs, retries=1)
    if result.signal not in {"yes", "no", "uncertain", "unrelated"}:
        result.signal = "unrelated"
    result.confidence = max(0.0, min(1.0, float(result.confidence or 0)))
    return result


def parse_signal(*, question_text: str, user_answer: str,
                 probe_label: str = "", evidence_texts: list[str] | None = None,
                 ) -> str:
    return parse_probe_answer(question_text=question_text, user_answer=user_answer,
                              probe_label=probe_label, evidence_texts=evidence_texts).signal
