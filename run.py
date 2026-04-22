from __future__ import annotations

import argparse
import json
from copy import deepcopy
from typing import Any
from uuid import uuid4

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

from medical_assistant.config import get_settings
from medical_assistant.graph.workflow import build_workflow

settings = get_settings()
workflow = build_workflow()

app = FastAPI(title=settings.app_name)

# 只用最轻量的内存会话，不依赖旧 schema
_SESSIONS: dict[str, dict[str, Any]] = {}

def _new_session(session_id: str | None = None) -> dict[str, Any]:
    sid = session_id or str(uuid4())
    return {
        "session_id": sid,
        "conversation_state": {"session_id": sid},
        "history": [],
    }

def _get_or_create_session(session_id: str | None = None) -> dict[str, Any]:
    if session_id and session_id in _SESSIONS:
        return _SESSIONS[session_id]

    session = _new_session(session_id)
    _SESSIONS[session["session_id"]] = session
    return session

def _extract_state(result: dict[str, Any], previous_state: dict[str, Any]) -> dict[str, Any]:
    for key in ("conversation_state", "state"):
        value = result.get(key)
        if isinstance(value, dict):
            return value
    return previous_state

def _extract_answer(result: dict[str, Any]) -> tuple[str, str]:
    response = result.get("response")

    if isinstance(response, dict):
        for key in ("content", "answer", "text", "message"):
            value = response.get(key)
            if isinstance(value, str) and value.strip():
                response_type = response.get("response_type") or response.get("type") or "answer"
                return value.strip(), str(response_type)

    if isinstance(response, str) and response.strip():
        return response.strip(), "answer"

    for key in ("answer", "final_answer", "content"):
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip(), "answer"

    # 最后兜底，至少把结果打出来，方便你继续调 workflow
    return json.dumps(result, ensure_ascii=False, indent=2), "debug"

def run_turn(session_id: str, question: str) -> dict[str, Any]:
    session = _get_or_create_session(session_id)

    payload = {
        "question": question,
        "conversation_state": deepcopy(session["conversation_state"]),
    }

    result = workflow.invoke(payload)

    new_state = _extract_state(result, session["conversation_state"])
    answer, response_type = _extract_answer(result)

    session["conversation_state"] = new_state
    session["history"].append({"role": "user", "content": question})
    session["history"].append(
        {"role": "assistant", "content": answer, "response_type": response_type}
    )

    return {
        "session_id": session["session_id"],
        "response_type": response_type,
        "answer": answer,
        "state": session["conversation_state"],
        "history": session["history"],
    }

def run_cli() -> None:
    session = _get_or_create_session()
    session_id = session["session_id"]

    print("Assistant CLI 已启动，输入 exit 退出。")
    print(f"session_id={session_id}")

    while True:
        try:
            question = input("\n[User]> ").strip()
        except KeyboardInterrupt:
            print("\n退出。")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        try:
            result = run_turn(session_id, question)
        except Exception as exc:
            print("\n[Error]")
            print(repr(exc))
            continue

        print("\n[Assistant]")
        print(result["answer"])

        print("\n[State]")
        print(json.dumps(result["state"], ensure_ascii=False, indent=2))

class SessionCreateResponse(BaseModel):
    session_id: str
    state: dict[str, Any]

class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str = Field(min_length=1)

@app.post("/api/session", response_model=SessionCreateResponse)
def create_session() -> SessionCreateResponse:
    session = _get_or_create_session()
    return SessionCreateResponse(
        session_id=session["session_id"],
        state=session["conversation_state"],
    )

@app.post("/api/chat")
def chat(payload: ChatRequest) -> dict[str, Any]:
    session = _get_or_create_session(payload.session_id)
    result = run_turn(session["session_id"], payload.message.strip())
    return result

def run_api() -> None:
    uvicorn.run(app, host=settings.api_host, port=settings.api_port, reload=False)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", nargs="?", choices=["cli", "api"], default="cli")
    args = parser.parse_args()

    if args.mode == "api":
        run_api()
    else:
        run_cli()

if __name__ == "__main__":
    main()