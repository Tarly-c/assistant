"""LLM 调用封装。API 优先，本地 Ollama 保底。

调用链：
  call_structured / call_text
    → 尝试 API（如果配了 api_key）
    → API 失败 → 自动降级到本地 Ollama
    → 本地也失败 → 返回默认值
"""
from __future__ import annotations

import json
import time
from functools import lru_cache
from typing import Type, TypeVar

from pydantic import BaseModel
from medical_assistant.config import get_settings

T = TypeVar("T", bound=BaseModel)

# ── LangChain 消息类型 ──
try:
    from langchain_core.messages import HumanMessage, SystemMessage
except Exception:
    HumanMessage = SystemMessage = None

# ── API 后端（OpenAI 兼容） ──
try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None

# ── 本地后端（Ollama） ──
try:
    from langchain_ollama import ChatOllama
except Exception:
    ChatOllama = None


# ═══════════════════════════════════════
# 模型实例管理
# ═══════════════════════════════════════

@lru_cache(maxsize=1)
def _api_model():
    """API 模型实例（DeepSeek / OpenAI 兼容）。"""
    if ChatOpenAI is None:
        return None
    s = get_settings()
    if not s.use_api:
        return None
    try:
        return ChatOpenAI(
            model=s.api_model,
            api_key=s.api_key,
            base_url=s.api_base_url,
            temperature=s.api_temperature,
            timeout=s.api_timeout,
            max_retries=1,
        )
    except Exception as e:
        print(f"[LLM] API model init failed: {e}")
        return None


@lru_cache(maxsize=1)
def _local_model():
    """本地 Ollama 模型实例。"""
    if ChatOllama is None:
        return None
    s = get_settings()
    try:
        return ChatOllama(
            model=s.local_model,
            temperature=s.local_temperature,
            num_ctx=4096,
        )
    except Exception as e:
        print(f"[LLM] Local model init failed: {e}")
        return None


def _to_lc(msgs: list[dict]) -> list:
    """dict 消息 → LangChain 消息对象。"""
    if HumanMessage is None:
        raise RuntimeError("langchain_core not installed")
    return [(SystemMessage if m.get("role") == "system" else HumanMessage)(
        content=m.get("content", "")) for m in msgs]


# ═══════════════════════════════════════
# 日志
# ═══════════════════════════════════════

def _log(tag: str, obj):
    print(f"\n[LLM {tag}]")
    if isinstance(obj, BaseModel):
        print(json.dumps(obj.model_dump(), ensure_ascii=False, indent=2))
    elif isinstance(obj, list):
        print(json.dumps(obj, ensure_ascii=False, indent=2))
    else:
        print(str(obj)[:500])


# ═══════════════════════════════════════
# 核心调用
# ═══════════════════════════════════════

def _try_structured(llm, schema: Type[T], lc_msgs) -> T | None:
    """尝试用某个 LLM 后端做结构化输出。"""
    try:
        bound = llm.with_structured_output(schema)
        result = bound.invoke(lc_msgs)
        return result if result is not None else None
    except Exception as e:
        print(f"  [structured failed] {type(llm).__name__}: {e}")
        return None


def _try_text(llm, lc_msgs) -> str | None:
    """尝试用某个 LLM 后端做纯文本输出。"""
    try:
        result = llm.invoke(lc_msgs)
        return result.content if result else None
    except Exception as e:
        print(f"  [text failed] {type(llm).__name__}: {e}")
        return None


def call_structured(schema: Type[T], messages: list[dict], *, retries: int = 1) -> T:
    """调 LLM 返回结构化 JSON。API 优先，本地保底。

    Args:
        schema: Pydantic 模型类
        messages: [{"role": "system"/"user", "content": "..."}]
        retries: 每个后端的重试次数
    """
    _log(f"{schema.__name__} payload", messages)
    lc = _to_lc(messages)
    s = get_settings()

    # 尝试顺序：API → 本地
    backends = []
    api = _api_model()
    local = _local_model()
    if api:
        backends.append(("API", api))
    if local:
        backends.append(("Local", local))

    for name, llm in backends:
        for attempt in range(retries + 1):
            t0 = time.time()
            result = _try_structured(llm, schema, lc)
            elapsed = time.time() - t0
            if result is not None:
                print(f"  [{name}] ok ({elapsed:.1f}s)")
                _log(f"{schema.__name__} result", result)
                return result
            if attempt < retries:
                print(f"  [{name}] retry {attempt+1}...")

    print(f"  [ALL FAILED] returning default {schema.__name__}")
    return schema()


def call_text(messages: list[dict]) -> str:
    """调 LLM 返回纯文本。API 优先，本地保底。"""
    _log("text payload", messages)
    lc = _to_lc(messages)

    backends = []
    api = _api_model()
    local = _local_model()
    if api:
        backends.append(("API", api))
    if local:
        backends.append(("Local", local))

    for name, llm in backends:
        t0 = time.time()
        result = _try_text(llm, lc)
        elapsed = time.time() - t0
        if result is not None:
            print(f"  [{name}] ok ({elapsed:.1f}s)")
            _log("text result", result)
            return result

    print(f"  [ALL FAILED] returning empty string")
    return ""


def which_backend() -> str:
    """返回当前使用的后端名称（调试用）。"""
    s = get_settings()
    api = _api_model()
    local = _local_model()
    if api:
        return f"API ({s.api_model} @ {s.api_base_url})"
    elif local:
        return f"Local ({s.local_model})"
    else:
        return "NONE (no backend available)"
