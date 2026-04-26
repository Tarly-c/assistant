from __future__ import annotations

"""LLM-based parser for replies to the active probe.

This module intentionally has no yes/no/uncertain word lists. It builds a
visible payload that can be printed or sent to the model for debugging.
"""

from medical_assistant.llm import invoke_structured
from medical_assistant.prompts import ANSWER_SIGNAL_PROMPT
from medical_assistant.schemas import AnswerSignalDraft

VALID_SIGNALS = {"yes", "no", "uncertain", "unrelated"}


def build_answer_signal_messages(
    user_reply: str,
    *,
    question_text: str,
    probe_label: str = "",
) -> list[dict]:
    return [
        {"role": "system", "content": ANSWER_SIGNAL_PROMPT},
        {
            "role": "user",
            "content": (
                "上一轮助手追问:\n"
                f"{question_text}\n\n"
                "当前用户回答:\n"
                f"{user_reply}\n\n"
                "当前 probe 标签:\n"
                f"{probe_label}\n"
            ),
        },
    ]


def parse_answer_signal(
    user_reply: str,
    *,
    question_text: str,
    probe_label: str = "",
) -> AnswerSignalDraft:
    if not (user_reply or "").strip() or not (question_text or "").strip():
        return AnswerSignalDraft(answer="unrelated", reason="empty reply or missing active question")

    result = invoke_structured(
        AnswerSignalDraft,
        build_answer_signal_messages(user_reply, question_text=question_text, probe_label=probe_label),
    )
    if result.answer not in VALID_SIGNALS:
        result.answer = "unrelated"
    return result
