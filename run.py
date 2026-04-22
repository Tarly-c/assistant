from __future__ import annotations

import argparse
import time
from functools import lru_cache
from threading import RLock
from typing import Any
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from medical_assistant.config import get_settings
from medical_assistant.graph.workflow import build_workflow


def _to_plain(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(k): _to_plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_plain(v) for v in value]
    if isinstance(value, tuple):
        return [_to_plain(v) for v in value]
    return value


def _as_dict(value: Any) -> dict[str, Any]:
    value = _to_plain(value)
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    value = _to_plain(value)
    return value if isinstance(value, list) else []


def _first_non_empty_text(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


class RuntimeSessionStore:
    def __init__(self) -> None:
        self._lock = RLock()
        self._sessions: dict[str, dict[str, Any]] = {}

    def create(self) -> dict[str, Any]:
        with self._lock:
            session_id = str(uuid4())
            now = time.time()
            record = {
                "session_id": session_id,
                "conversation_state": {},
                "messages": [],
                "created_at": now,
                "updated_at": now,
            }
            self._sessions[session_id] = record
            return record

    def get_or_create(self, session_id: str | None = None) -> dict[str, Any]:
        with self._lock:
            if session_id and session_id in self._sessions:
                return self._sessions[session_id]
            return self.create()

    def save_turn(
        self,
        session_id: str,
        *,
        conversation_state: dict[str, Any],
        message_record: dict[str, Any],
    ) -> dict[str, Any]:
        with self._lock:
            record = self._sessions[session_id]
            record["conversation_state"] = conversation_state
            record["messages"].append(message_record)
            record["updated_at"] = time.time()
            self._sessions[session_id] = record
            return record

    def get(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            return self._sessions[session_id]


STORE = RuntimeSessionStore()


@lru_cache(maxsize=1)
def get_workflow():
    return build_workflow()


def _extract_response_payload(result: dict[str, Any]) -> dict[str, Any]:
    result = _as_dict(result)

    conversation_state = _as_dict(result.get("conversation_state"))
    response = _as_dict(result.get("response"))
    confidence = _as_dict(result.get("confidence")) or _as_dict(
        conversation_state.get("confidence")
    )
    safety = _as_dict(response.get("safety")) or _as_dict(result.get("safety"))
    local_result = _as_dict(result.get("local_result"))
    web_result = _as_dict(result.get("web_result"))
    plan = _as_dict(result.get("plan"))
    clarify = _as_dict(result.get("clarify"))

    matched_topics = _as_list(response.get("matched_topics"))
    if not matched_topics:
        matched_topics = _as_list(result.get("candidate_topics"))

    content = _first_non_empty_text(
        response.get("content"),
        response.get("answer"),
        response.get("message"),
        response.get("text"),
        clarify.get("question"),
        conversation_state.get("last_clarify_question"),
        result.get("answer"),
    )

    phase = str(conversation_state.get("phase") or "")
    response_type = str(response.get("response_type") or "").strip()

    if not response_type:
        if safety.get("level") == "high" or phase == "SAFETY_ESCALATION":
            response_type = "safety"
        elif phase == "NEEDS_CLARIFICATION" or conversation_state.get(
            "last_clarify_question"
        ):
            response_type = "clarification"
        else:
            response_type = "answer"

    return {
        "response_type": response_type,
        "answer": content,
        "matched_topics": matched_topics,
        "confidence": confidence,
        "safety": safety,
        "plan": plan,
        "local_result": local_result,
        "pubmed_result": web_result,
        "state": conversation_state,
        "raw_result": result,
    }


def run_one_turn(session_id: str | None, question: str) -> dict[str, Any]:
    record = STORE.get_or_create(session_id)
    workflow = get_workflow()

    result = workflow.invoke(
        {
            "question": question,
            "conversation_state": record.get("conversation_state", {}),
        }
    )
    result = _as_dict(result)

    new_state = _as_dict(result.get("conversation_state")) or record.get(
        "conversation_state", {}
    )

    payload = _extract_response_payload(result)

    STORE.save_turn(
        record["session_id"],
        conversation_state=new_state,
        message_record={
            "user": question,
            "assistant": payload["answer"],
            "response_type": payload["response_type"],
            "state": payload["state"],
            "confidence": payload["confidence"],
            "created_at": time.time(),
        },
    )

    return {
        "session_id": record["session_id"],
        "response_type": payload["response_type"],
        "answer": payload["answer"],
        "matched_topics": payload["matched_topics"],
        "confidence": payload["confidence"],
        "safety": payload["safety"],
        "plan": payload["plan"],
        "local_result": payload["local_result"],
        "pubmed_result": payload["pubmed_result"],
        "state": payload["state"],
        "history_count": len(STORE.get(record["session_id"])["messages"]),
    }


def run_cli(show_debug: bool = False) -> None:
    record = STORE.create()
    session_id = record["session_id"]

    print("Medical Assistant CLI 已启动。输入 exit 退出。")
    print(f"session_id={session_id}")

    while True:
        try:
            question = input("\n[User]> ").strip()
        except KeyboardInterrupt:
            print("\n退出。")
            break

        if not question:
            continue
        if question.lower() == "exit":
            break

        try:
            payload = run_one_turn(session_id, question)
        except Exception as exc:
            print("\n[Error]")
            print(str(exc))
            continue

        print("\n[Assistant]")
        print(payload["answer"] or "(empty response)")

        print("\n[State]")
        state = payload.get("state", {}) or {}
        print(
            {
                "phase": state.get("phase"),
                "turn_index": state.get("turn_index"),
                "last_clarify_question": state.get("last_clarify_question"),
                "confidence": payload.get("confidence", {}),
            }
        )

        if show_debug:
            print("\n[Debug]")
            print(
                {
                    "response_type": payload.get("response_type"),
                    "matched_topics_count": len(payload.get("matched_topics", [])),
                    "local_result_keys": list((payload.get("local_result") or {}).keys()),
                    "pubmed_result_keys": list((payload.get("pubmed_result") or {}).keys()),
                }
            )


settings = get_settings()
web_app = FastAPI(title=settings.app_name)
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SessionCreateResponse(BaseModel):
    session_id: str
    state: dict[str, Any]


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str = Field(min_length=1)


@web_app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "app_name": settings.app_name,
        "api_host": settings.api_host,
        "api_port": settings.api_port,
    }


@web_app.post("/api/session", response_model=SessionCreateResponse)
def create_session() -> SessionCreateResponse:
    record = STORE.create()
    return SessionCreateResponse(
        session_id=record["session_id"],
        state=record["conversation_state"],
    )


@web_app.post("/api/chat")
def chat(payload: ChatRequest) -> dict[str, Any]:
    question = payload.message.strip()
    if not question:
        raise HTTPException(status_code=400, detail="message 不能为空")

    try:
        return run_one_turn(payload.session_id, question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def run_api() -> None:
    uvicorn.run(
        web_app,
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        nargs="?",
        choices=["cli", "api"],
        default="cli",
        help="cli 或 api",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="CLI 模式下打印调试信息",
    )
    args = parser.parse_args()

    if args.mode == "api":
        run_api()
    else:
        run_cli(show_debug=args.debug)


if __name__ == "__main__":
    main()
