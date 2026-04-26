from __future__ import annotations

"""LLM-based parsing for answers to the active probe.

There is no yes/no/uncertain word table here. The parser builds a visible JSON
payload and, when invoked by the app, asks the configured LLM to classify the
reply relative to the previous assistant question.
"""

import json
import os
from typing import Any

from medical_assistant.llm import invoke_structured
from medical_assistant.schemas import ProbeAnswerParse


SYSTEM_PROMPT = """
你是问诊工作流中的回答解析器。你的任务不是诊断疾病，而是判断用户这句话是否回答了上一轮追问。
只允许输出结构化字段 signal/confidence/evidence/new_observations。
signal 只能是 yes、no、uncertain、unrelated 之一：
- yes: 用户明确表示上一轮追问中的情况符合自己；
- no: 用户明确否定上一轮追问中的情况，或描述了相反情况；
- uncertain: 用户在回答这个问题，但表示不确定、说不准、只有一点点；
- unrelated: 用户没有回答上一轮追问，而是在换话题或提供无法判断的信息。
""".strip()


def build_probe_answer_payload(
    *,
    question_text: str,
    user_answer: str,
    probe_label: str = "",
    evidence_texts: list[str] | None = None,
) -> list[dict[str, str]]:
    data: dict[str, Any] = {
        "previous_question": question_text,
        "probe_label": probe_label,
        "probe_evidence_texts": evidence_texts or [],
        "user_answer": user_answer,
        "output_schema": {
            "signal": "yes | no | uncertain | unrelated",
            "confidence": "0.0-1.0",
            "evidence": "短中文理由",
            "new_observations": "用户额外提供的新症状片段数组",
        },
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(data, ensure_ascii=False, indent=2)},
    ]


def parse_probe_answer(
    *,
    question_text: str,
    user_answer: str,
    probe_label: str = "",
    evidence_texts: list[str] | None = None,
) -> ProbeAnswerParse:
    messages = build_probe_answer_payload(
        question_text=question_text,
        user_answer=user_answer,
        probe_label=probe_label,
        evidence_texts=evidence_texts,
    )

    if os.getenv("MEDICAL_ASSISTANT_DEBUG_LLM_PAYLOADS", "").lower() in {"1", "true", "yes"}:
        print("[LLM payload] answer_parser")
        print(json.dumps(messages, ensure_ascii=False, indent=2))

    result = invoke_structured(ProbeAnswerParse, messages, retries=1)
    if result.signal not in {"yes", "no", "uncertain", "unrelated"}:
        result.signal = "unrelated"
    result.confidence = max(0.0, min(1.0, float(result.confidence or 0.0)))
    return result


def parse_probe_answer_signal(
    *,
    question_text: str,
    user_answer: str,
    probe_label: str = "",
    evidence_texts: list[str] | None = None,
) -> str:
    return parse_probe_answer(
        question_text=question_text,
        user_answer=user_answer,
        probe_label=probe_label,
        evidence_texts=evidence_texts,
    ).signal
