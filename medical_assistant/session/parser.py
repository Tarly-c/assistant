"""LLM 回答分类：用户回答是 yes / no / uncertain / unrelated。"""
from __future__ import annotations

import json

from medical_assistant.llm import call_structured
from medical_assistant.prompts import PARSE_ANSWER
from medical_assistant.schemas import ParsedAnswer


def parse_answer(*, probe_text: str, user_input: str,
                 probe_label: str = "") -> ParsedAnswer:
    """调用 LLM 判断用户回答与上一轮追问的关系。"""
    data = {"previous_question": probe_text,
            "probe_label": probe_label, "user_answer": user_input}
    messages = [
        {"role": "system", "content": PARSE_ANSWER},
        {"role": "user", "content": json.dumps(data, ensure_ascii=False, indent=2)},
    ]
    result = call_structured(ParsedAnswer, messages, retries=1)
    if result.signal not in {"yes", "no", "uncertain", "unrelated"}:
        result.signal = "unrelated"
    result.confidence = max(0.0, min(1.0, float(result.confidence or 0)))
    return result
