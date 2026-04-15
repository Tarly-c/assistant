from __future__ import annotations

import os
import re
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import app as agent_graph

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "web_static"
SESSION_TTL_SECONDS = int(os.getenv("WEB_SESSION_TTL_SECONDS", "7200"))
MAX_CONTEXT_TURNS = int(os.getenv("WEB_CONTEXT_TURNS", "3"))

web_app = FastAPI(title="Medical Assistant Web UI", version="1.0.0")
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    web_app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

SESSIONS: dict[str, dict[str, Any]] = {}
TITLE_RE = re.compile(r"^\s*#\s+(.+?)\s*$", re.MULTILINE)
WHITESPACE_RE = re.compile(r"\s+")


class SessionCreateResponse(BaseModel):
    session_id: str
    created_at: float


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Browser session id")
    message: str = Field(..., min_length=1, description="User message")


class CloseSessionRequest(BaseModel):
    session_id: str


INTENT_LABELS = {
    "treatment": "治疗 / 处理建议",
    "cause": "可能原因",
    "symptom": "症状解释",
    "diagnosis": "诊断相关",
    "general": "一般医学信息",
}


def _compact(text: str | None) -> str:
    return WHITESPACE_RE.sub(" ", (text or "")).strip()


def _cleanup_sessions() -> None:
    now = time.time()
    expired = [
        sid
        for sid, session in SESSIONS.items()
        if now - float(session.get("updated_at", session.get("created_at", now))) > SESSION_TTL_SECONDS
    ]
    for sid in expired:
        SESSIONS.pop(sid, None)


def _new_session() -> dict[str, Any]:
    now = time.time()
    return {
        "created_at": now,
        "updated_at": now,
        "messages": [],
        "active": True,
    }


def _get_session(session_id: str) -> dict[str, Any]:
    _cleanup_sessions()
    session = SESSIONS.get(session_id)
    if not session or not session.get("active"):
        raise HTTPException(status_code=404, detail="会话不存在或已结束")
    return session


def _extract_local_title(source: str, snippet: str) -> str:
    match = TITLE_RE.search(snippet or "")
    if match:
        return _compact(match.group(1))
    source_name = Path(source).stem if source else "本地文档"
    return source_name.replace("_", " ").strip() or "本地文档"


def _build_contextual_question(session: dict[str, Any], message: str) -> str:
    history = session.get("messages", [])[-MAX_CONTEXT_TURNS:]
    if not history:
        return message

    lines = [
        "以下是最近对话，请结合上下文理解当前问题，但回答时只直接回答当前问题。",
    ]
    for item in history:
        user_text = _compact(item.get("user"))
        assistant_text = _compact(item.get("answer"))
        if user_text:
            lines.append(f"用户：{user_text}")
        if assistant_text:
            lines.append(f"助手：{assistant_text}")
    lines.append(f"当前问题：{message}")
    return "\n".join(lines)


def _clean_plan(plan: dict[str, Any]) -> dict[str, Any]:
    local_query = _compact(plan.get("local_query_en"))
    pubmed_queries: list[str] = []
    seen: set[str] = set()
    for item in plan.get("pubmed_queries", []) or []:
        cleaned = _compact(item)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            pubmed_queries.append(cleaned)

    payload: dict[str, Any] = {
        "问题意图": INTENT_LABELS.get(plan.get("intent"), "一般医学信息"),
    }
    if local_query:
        payload["本地知识库检索词"] = local_query
    if pubmed_queries:
        payload["PubMed 检索词"] = pubmed_queries
    return payload


def _clean_local_result(local_result: dict[str, Any]) -> dict[str, Any]:
    hits = []
    for idx, hit in enumerate(local_result.get("hits", []) or [], start=1):
        source = str(hit.get("source") or "")
        snippet = (hit.get("snippet") or "").strip()
        hits.append(
            {
                "序号": idx,
                "标题": _extract_local_title(source, snippet),
                "文件": Path(source).as_posix() if source else "",
                "Chunk": hit.get("chunk_id"),
                "分数": round(float(hit.get("score", 0.0)), 4),
                "摘要": snippet,
            }
        )

    return {
        "是否足够": bool(local_result.get("enough")),
        "最高分": round(float(local_result.get("score", 0.0)), 4),
        "原因": local_result.get("reason"),
        "命中": hits,
    }


def _clean_pubmed_result(web_result: dict[str, Any]) -> dict[str, Any]:
    hits = []
    for idx, hit in enumerate(web_result.get("hits", []) or [], start=1):
        hits.append(
            {
                "序号": idx,
                "PMID": str(hit.get("pmid") or ""),
                "标题": _compact(hit.get("title")),
                "期刊": _compact(hit.get("journal")),
                "日期": _compact(hit.get("pubdate")),
                "重排分数": round(float(hit.get("rerank_score", 0.0)), 4),
                "摘要": (hit.get("snippet") or "").strip(),
            }
        )

    return {
        "查询": _compact(web_result.get("query")),
        "命中": hits,
    }


@web_app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@web_app.post("/api/sessions", response_model=SessionCreateResponse)
def create_session() -> SessionCreateResponse:
    _cleanup_sessions()
    session_id = uuid.uuid4().hex
    session = _new_session()
    SESSIONS[session_id] = session
    return SessionCreateResponse(session_id=session_id, created_at=session["created_at"])


@web_app.post("/api/sessions/close")
def close_session(payload: CloseSessionRequest) -> dict[str, Any]:
    session = SESSIONS.get(payload.session_id)
    if session:
        session["active"] = False
        session["updated_at"] = time.time()
        SESSIONS.pop(payload.session_id, None)
    return {"ok": True}


@web_app.post("/api/chat")
def chat(payload: ChatRequest) -> dict[str, Any]:
    session = _get_session(payload.session_id)
    user_message = payload.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="消息不能为空")

    contextual_question = _build_contextual_question(session, user_message)
    result = agent_graph.invoke({"question": contextual_question})

    answer = (result.get("answer") or "").strip()
    cleaned_plan = _clean_plan(result.get("plan", {}))
    cleaned_local = _clean_local_result(result.get("local_result", {}))
    cleaned_pubmed = _clean_pubmed_result(result.get("web_result", {}))

    message_record = {
        "user": user_message,
        "answer": answer,
        "plan": cleaned_plan,
        "local": cleaned_local,
        "pubmed": cleaned_pubmed,
        "created_at": time.time(),
    }
    session["messages"].append(message_record)
    session["updated_at"] = time.time()

    return {
        "session_id": payload.session_id,
        "question": user_message,
        "answer": answer,
        "plan": cleaned_plan,
        "plan_raw": result.get("plan", {}),
        "local_result": cleaned_local,
        "pubmed_result": cleaned_pubmed,
        "history_count": len(session["messages"]),
    }


@web_app.get("/")
def index() -> FileResponse:
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="index.html 不存在")
    return FileResponse(index_file)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("web_server:web_app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
