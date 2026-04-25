"""LLM 工具函数：结构化输出 + 纯文本输出。

The case demo can run without an Ollama server. If LangChain/Ollama is not
installed or not reachable, structured calls return an empty schema instance and
normalize.py will fall back to deterministic keyword extraction.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Type, TypeVar

from pydantic import BaseModel

from medical_assistant.config import get_settings

try:  # Optional at runtime for the local demo.
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_ollama import ChatOllama
except Exception:  # pragma: no cover - depends on local environment
    HumanMessage = None
    SystemMessage = None
    ChatOllama = None


T = TypeVar("T", bound=BaseModel)


@lru_cache(maxsize=1)
def _get_chat_model():
    if ChatOllama is None:
        raise RuntimeError("langchain_ollama is not installed")
    settings = get_settings()
    return ChatOllama(
        model=settings.chat_model,
        temperature=settings.temperature,
        num_ctx=4096,
    )


def _to_langchain_messages(raw_messages: list[dict]) -> list:
    if HumanMessage is None or SystemMessage is None:
        raise RuntimeError("langchain_core is not installed")
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
    retries: int = 1,
) -> T:
    """让 LLM 输出符合 schema 的 JSON；失败时返回默认空实例。"""

    last_err: Exception | None = None
    try:
        llm = _get_chat_model().with_structured_output(schema)
        lc_msgs = _to_langchain_messages(messages)
    except Exception as exc:
        return schema()

    for _ in range(retries + 1):
        try:
            result = llm.invoke(lc_msgs)
            if result is not None:
                return result
        except Exception as exc:  # pragma: no cover - depends on Ollama server
            last_err = exc
    if last_err:
        print(f"[LLM] structured output failed: {last_err}")
    return schema()


def invoke_text(messages: list[dict]) -> str:
    try:
        llm = _get_chat_model()
        lc_msgs = _to_langchain_messages(messages)
        result = llm.invoke(lc_msgs)
        return result.content if result else ""
    except Exception:
        return ""
