from __future__ import annotations

from typing import Any

from medical_assistant.schemas.input import NormalizedInput, QueryKeyword, SearchQuery
from medical_assistant.schemas.state import ConversationState
from medical_assistant.services.llm import invoke_structured

NORMALIZE_PROMPT = """
你是医学检索预处理器。
只做以下事情：
1. 把最新用户输入忠实地转述成自然、准确的英文。
2. 提取少量对检索有帮助的关键词。
3. 组织成一个主查询和少量备选查询。

要求：
- 不要诊断。
- 不要补充用户没说过的信息。
- 不要依赖固定病例模板。
- translated_en 必须是适合检索的英文表达。
- keywords 控制在 3~8 个，尽量短。
- queries 至少 2 条：一条完整英文查询，一条关键词重组查询。
- follow_up_hints 只写仍缺少但普适的检索维度，例如持续时间、严重程度、诱因、伴随表现；最多 3 条。
- 只输出结构化结果。
""".strip()


def _fallback_keywords(text: str) -> list[QueryKeyword]:
    tokens = [item.strip(" ,.;:!?()[]{}\"'\n\t") for item in text.split()]
    values = [token for token in tokens if token][:6]
    return [
        QueryKeyword(
            text=item,
            category="user_term",
            normalized_en=item,
            confidence=0.3,
            source="system",
        )
        for item in values
    ]


def _fallback_normalized(question: str) -> NormalizedInput:
    normalized = NormalizedInput(
        original_text=question,
        translated_en=question,
        keywords=_fallback_keywords(question),
        queries=[
            SearchQuery(text=question, channel="local", purpose="primary", weight=1.0),
        ],
        follow_up_hints=[],
    )
    return normalized.prepare_for_runtime()


def normalize_node(state: dict[str, Any]) -> dict[str, Any]:
    question = state["question"]
    conversation_state = ConversationState.model_validate(state.get("conversation_state") or {})

    try:
        normalized = invoke_structured(
            NormalizedInput,
            [
                {"role": "system", "content": NORMALIZE_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "已有结构化上下文：\n"
                        f"{conversation_state.model_dump_json(indent=2, ensure_ascii=False)}\n\n"
                        "最新用户输入：\n"
                        f"{question}"
                    ),
                },
            ],
        )
        normalized.original_text = normalized.original_text or question
        normalized = normalized.prepare_for_runtime()
    except Exception:
        normalized = _fallback_normalized(question)

    return {"normalized": normalized.model_dump(mode="json")}
