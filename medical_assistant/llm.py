"""LLM 调用封装。始终打印 payload。"""
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


def _to_lc(msgs: list[dict]) -> list:
    return [(SystemMessage if m.get("role") == "system" else HumanMessage)(
        content=m.get("content", "")) for m in msgs]


def _log(tag: str, obj):
    print(f"\n[LLM {tag}]")
    if isinstance(obj, BaseModel):
        print(json.dumps(obj.model_dump(), ensure_ascii=False, indent=2))
    elif isinstance(obj, list):
        print(json.dumps(obj, ensure_ascii=False, indent=2))
    else:
        print(str(obj)[:500])


def call_structured(schema: Type[T], messages: list[dict], *, retries: int = 1) -> T:
    """调 LLM 返回结构化 JSON。"""
    _log(f"{schema.__name__} payload", messages)
    try:
        llm = _model().with_structured_output(schema)
        lc = _to_lc(messages)
    except Exception as e:
        print(f"[LLM setup failed] {e}")
        return schema()
    for _ in range(retries + 1):
        try:
            r = llm.invoke(lc)
            if r is not None:
                _log(f"{schema.__name__} result", r)
                return r
        except Exception as e:
            print(f"[LLM call failed] {e}")
    return schema()


def call_text(messages: list[dict]) -> str:
    """调 LLM 返回纯文本。"""
    _log("text payload", messages)
    try:
        r = _model().invoke(_to_lc(messages))
        text = r.content if r else ""
        _log("text result", text)
        return text
    except Exception as e:
        print(f"[LLM text failed] {e}")
        return ""
