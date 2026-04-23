"""respond 节点：answer / clarify + 路由。无 safety。"""
from __future__ import annotations

from medical_assistant.graph.state import GraphState
from medical_assistant.llm import invoke_structured
from medical_assistant.prompts import ANSWER_PROMPT, CLARIFY_PROMPT
from medical_assistant.schemas import AnswerDraft, ClarifyDraft


def route_after_retrieve(state: GraphState) -> str:
    """检索分数太低就追问，否则直接回答。"""
    if not state.get("enough", False) and state.get("best_score", 0.0) < 0.35:
        return "clarify"
    return "answer"


def answer_node(state: GraphState) -> dict:
    question = state.get("question", "")
    hits = state.get("hits", [])

    parts: list[str] = []
    for i, hit in enumerate(hits[:5], 1):
        title = hit.get("title", "未知来源")
        snippet = hit.get("snippet", "")
        parts.append(f"[{i}] {title}\n{snippet}")

    context = "\n\n".join(parts) if parts else "（没有找到相关资料）"

    result = invoke_structured(
        AnswerDraft,
        [
            {"role": "system", "content": ANSWER_PROMPT},
            {"role": "user", "content": f"用户问题: {question}\n\n【参考资料】\n{context}"},
        ],
    )

    return {
        "response_type": "answer",
        "answer": result.answer or "抱歉，暂时无法回答这个问题，建议咨询专业医生。",
        "sources": result.sources_used,
        "confidence": round(state.get("best_score", 0.0), 4),
        "phase": "ANSWERED",
        "turn_index": state.get("turn_index", 0) + 1,
    }


def clarify_node(state: GraphState) -> dict:
    question = state.get("question", "")

    result = invoke_structured(
        ClarifyDraft,
        [
            {"role": "system", "content": CLARIFY_PROMPT},
            {"role": "user", "content": f"用户问题: {question}"},
        ],
    )

    return {
        "response_type": "clarification",
        "answer": result.question or "能否提供更多细节？比如症状持续多久了、具体在哪个部位？",
        "confidence": round(state.get("best_score", 0.0), 4),
        "phase": "NEEDS_CLARIFICATION",
        "turn_index": state.get("turn_index", 0) + 1,
    }
