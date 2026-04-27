"""LLM 回答分类。"""
from __future__ import annotations
import json
from medical_assistant.llm import call_structured
from medical_assistant.prompts import PARSE_ANSWER
from medical_assistant.schemas import ParsedAnswer


def parse_answer(*, probe_text: str, user_input: str,
                 probe_label: str = "") -> ParsedAnswer:
    data = {"previous_question": probe_text,
            "probe_label": probe_label, "user_answer": user_input}
    r = call_structured(ParsedAnswer, [
        {"role": "system", "content": PARSE_ANSWER},
        {"role": "user", "content": json.dumps(data, ensure_ascii=False, indent=2)},
    ], retries=1)
    if r.signal not in {"yes", "no", "uncertain", "unrelated"}:
        r.signal = "unrelated"
    r.confidence = max(0.0, min(1.0, float(r.confidence or 0)))
    return r
