"""LLM 调用封装。始终打印 payload 和结果。"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Type, TypeVar

from pydantic import BaseModel
from medical_assistant.config import get_settings

try:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_ollama import ChatOllama
except Exception:
    HumanMessage = SystemMessage = ChatOllama = None

T = TypeVar("T", bound=BaseModel)


@lru_cache(maxsize=1)
def _model():
    if ChatOllama is None:
        raise RuntimeError("langchain_ollama not installed")
    s = get_settings()
    return ChatOllama(model=s.chat_model, temperature=s.temperature, num_ctx=4096)


def _to_lc(messages: list[dict]) -> list:
    """dict 消息列表 → LangChain 消息对象列表。"""
    out = []
    for m in messages:
        cls = SystemMessage if m.get("role") == "system" else HumanMessage
        out.append(cls(content=m.get("content", "")))
    return out


def _print_payload(tag: str, messages: list[dict]) -> None:
    print(f"\n[LLM {tag} payload]")
    print(json.dumps(messages, ensure_ascii=False, indent=2))


def _print_result(tag: str, obj) -> None:
    print(f"\n[LLM {tag} result]")
    if isinstance(obj, BaseModel):
        print(json.dumps(obj.model_dump(), ensure_ascii=False, indent=2))
    else:
        print(obj)


def call_structured(schema: Type[T], messages: list[dict], *, retries: int = 1) -> T:
    """调用 LLM 返回结构化 JSON。失败时返回空 schema 实例。"""
    _print_payload(schema.__name__, messages)
    try:
        llm = _model().with_structured_output(schema)
        lc = _to_lc(messages)
    except Exception as exc:
        print(f"[LLM setup failed] {exc}")
        return schema()

    for _ in range(retries + 1):
        try:
            result = llm.invoke(lc)
            if result is not None:
                _print_result(schema.__name__, result)
                return result
        except Exception as exc:
            print(f"[LLM call failed] {exc}")
    return schema()


def call_text(messages: list[dict]) -> str:
    """调用 LLM 返回纯文本。"""
    _print_payload("text", messages)
    try:
        result = _model().invoke(_to_lc(messages))
        text = result.content if result else ""
        _print_result("text", text)
        return text
    except Exception as exc:
        print(f"[LLM text failed] {exc}")
        return ""
