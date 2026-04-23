"""normalize 节点：1 次 LLM 调用，提取 3 个字段。"""
from __future__ import annotations

from medical_assistant.graph.state import GraphState
from medical_assistant.llm import invoke_structured
from medical_assistant.prompts import NORMALIZE_PROMPT
from medical_assistant.schemas import NormalizedInput


def normalize_node(state: GraphState) -> dict:
    question = state.get("question", "")
    history = state.get("conversation_history", [])

    # 如果有历史，把最近 2 轮拼进 prompt 给模型上下文
    history_text = ""
    if history:
        recent = history[-4:]  # 最近 2 轮（user+assistant 各一条）
        lines = [f'{m["role"]}: {m["content"]}' for m in recent]
        history_text = "对话历史:\n" + "\n".join(lines) + "\n\n"

    result = invoke_structured(
        NormalizedInput,
        [
            {"role": "system", "content": NORMALIZE_PROMPT},
            {"role": "user", "content": f"{history_text}当前问题: {question}"},
        ],
    )

    return {
        "query_en": result.query_en or question,
        "intent": result.intent or "general",
        "key_terms_en": result.key_terms_en or [],
    }
