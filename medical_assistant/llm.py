"""LLM 调用封装。不依赖 Ollama 也能运行（返回空 schema 实例）。"""
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
def _get_chat_model():
    if ChatOllama is None:
        raise RuntimeError("langchain_ollama is not installed")
    s = get_settings()
    return ChatOllama(model=s.chat_model, temperature=s.temperature, num_ctx=4096)


def _to_lc(raw: list[dict]) -> list:
    out = []
    for m in raw:
        if m.get("role") == "system":
            out.append(SystemMessage(content=m.get("content", "")))
        else:
            out.append(HumanMessage(content=m.get("content", "")))
    return out


def _debug() -> bool:
    try:
        return bool(get_settings().debug_llm_payloads)
    except Exception:
        return False


def print_llm_payload(*, schema_name: str, messages: list[dict]) -> None:
    print("\n[LLM payload]")
    print(json.dumps({"schema": schema_name, "messages": messages}, ensure_ascii=False, indent=2))


def invoke_structured(schema: Type[T], messages: list[dict], *, retries: int = 1) -> T:
    if _debug():
        print_llm_payload(schema_name=schema.__name__, messages=messages)
    try:
        llm = _get_chat_model().with_structured_output(schema)
        lc_msgs = _to_lc(messages)
    except Exception as exc:
        if _debug():
            print(f"[LLM setup failed] {exc}")
        return schema()
    last_err = None
    for _ in range(retries + 1):
        try:
            result = llm.invoke(lc_msgs)
            if result is not None:
                if _debug():
                    print(f"\n[LLM result] {json.dumps(result.model_dump(), ensure_ascii=False, indent=2)}")
                return result
        except Exception as exc:
            last_err = exc
    if last_err:
        print(f"[LLM] structured output failed: {last_err}")
    return schema()


def invoke_text(messages: list[dict]) -> str:
    if _debug():
        print_llm_payload(schema_name="text", messages=messages)
    try:
        result = _get_chat_model().invoke(_to_lc(messages))
        text = result.content if result else ""
        if _debug():
            print(f"\n[LLM text] {text}")
        return text
    except Exception as exc:
        if _debug():
            print(f"[LLM text failed] {exc}")
        return ""
