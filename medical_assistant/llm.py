"""LLM 工具函数：结构化输出 + 纯文本输出。"""
from __future__ import annotations

from functools import lru_cache
from typing import Type, TypeVar

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from medical_assistant.config import get_settings

T = TypeVar("T", bound=BaseModel)


@lru_cache(maxsize=1)
def _get_chat_model() -> ChatOllama:
    settings = get_settings()
    return ChatOllama(
        model=settings.chat_model,
        temperature=0.1,
        num_ctx=4096,
    )


def _to_langchain_messages(raw_messages: list[dict]) -> list:
    """把 {"role": ..., "content": ...} 转成 langchain 消息对象。"""
    out = []
    for m in raw_messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            out.append(SystemMessage(content=content))
        else:
            out.append(HumanMessage(content=content))
    return out


def invoke_structured(
    schema: Type[T],
    messages: list[dict],
    *,
    retries: int = 2,
) -> T:
    """让 LLM 输出符合 schema 的 JSON，失败时重试并返回默认值。"""
    llm = _get_chat_model().with_structured_output(schema)
    lc_msgs = _to_langchain_messages(messages)

    last_err = None
    for _ in range(retries + 1):
        try:
            result = llm.invoke(lc_msgs)
            if result is not None:
                return result
        except Exception as exc:
            last_err = exc

    # 全部失败，返回默认空实例
    print(f"[LLM] structured output failed after {retries+1} attempts: {last_err}")
    return schema()


def invoke_text(messages: list[dict]) -> str:
    """纯文本调用，用于不需要结构化输出的场景。"""
    llm = _get_chat_model()
    lc_msgs = _to_langchain_messages(messages)
    result = llm.invoke(lc_msgs)
    return result.content if result else ""
