from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain_ollama import ChatOllama

from medical_assistant.config import get_settings


@lru_cache(maxsize=1)
def get_chat_model() -> ChatOllama:
    settings = get_settings()
    return ChatOllama(
        model=settings.chat_model,
        temperature=settings.temperature,
    )


def invoke_structured(schema: type, messages: list[dict[str, Any]]):
    model = get_chat_model().with_structured_output(schema)
    return model.invoke(messages)


def invoke_text(messages: list[dict[str, Any]]) -> str:
    result = get_chat_model().invoke(messages)
    return getattr(result, "content", str(result))
