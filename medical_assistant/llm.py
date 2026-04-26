"""LLM tools: structured output and text output.

The case demo can run without an Ollama server. If LangChain/Ollama is not
installed or not reachable, structured calls return an empty schema instance.
When `MEDICAL_ASSISTANT_DEBUG_LLM_PAYLOADS=true`, every LLM payload and parsed
result is printed to the terminal for interactive debugging.
"""

from __future__ import annotations

import json
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


def _debug_enabled() -> bool:
    try:
        return bool(get_settings().debug_llm_payloads)
    except Exception:
        return False


def print_llm_payload(*, schema_name: str, messages: list[dict]) -> None:
    print("\n[LLM payload]")
    print(json.dumps({"schema": schema_name, "messages": messages}, ensure_ascii=False, indent=2))


def _print_structured_result(result: BaseModel | None) -> None:
    print("\n[LLM structured result]")
    if result is None:
        print("null")
    else:
        print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))


def invoke_structured(
    schema: Type[T],
    messages: list[dict],
    *,
    retries: int = 1,
) -> T:
    """Ask the LLM for schema-conforming JSON; return empty schema on failure."""

    if _debug_enabled():
        print_llm_payload(schema_name=schema.__name__, messages=messages)

    last_err: Exception | None = None
    try:
        llm = _get_chat_model().with_structured_output(schema)
        lc_msgs = _to_langchain_messages(messages)
    except Exception as exc:
        if _debug_enabled():
            print(f"[LLM setup failed] {exc}")
        return schema()

    for _ in range(retries + 1):
        try:
            result = llm.invoke(lc_msgs)
            if result is not None:
                if _debug_enabled():
                    _print_structured_result(result)
                return result
        except Exception as exc:  # pragma: no cover - depends on local server
            last_err = exc

    if last_err:
        print(f"[LLM] structured output failed: {last_err}")
    return schema()


def invoke_text(messages: list[dict]) -> str:
    if _debug_enabled():
        print_llm_payload(schema_name="text", messages=messages)
    try:
        llm = _get_chat_model()
        lc_msgs = _to_langchain_messages(messages)
        result = llm.invoke(lc_msgs)
        text = result.content if result else ""
        if _debug_enabled():
            print("\n[LLM text result]")
            print(text)
        return text
    except Exception as exc:
        if _debug_enabled():
            print(f"[LLM text failed] {exc}")
        return ""
