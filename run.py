"""启动入口：CLI 或 API。"""
from __future__ import annotations

import argparse
import time
from typing import Any
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from medical_assistant.config import get_settings
from medical_assistant.graph.workflow import build_workflow

_workflow = None


def get_workflow():
    global _workflow
    if _workflow is None:
        _workflow = build_workflow()
    return _workflow


_sessions: dict[str, dict] = {}


def _get_or_create_session(session_id: str | None) -> dict:
    if session_id and session_id in _sessions:
        return _sessions[session_id]
    sid = session_id or str(uuid4())
    record = {
        "session_id": sid,
        "history": [],
        "turn_index": 0,
        "created_at": time.time(),
        "case_memory": {},
    }
    _sessions[sid] = record
    return record


def run_one_turn(session_id: str | None, question: str) -> dict[str, Any]:
    session = _get_or_create_session(session_id)
    wf = get_workflow()
    result = wf.invoke(
        {
            "question": question,
            "conversation_history": session["history"],
            "turn_index": session["turn_index"],
            "case_memory": session.get("case_memory", {}),
        }
    )
    answer_text = result.get("answer", "")
    session["history"].append({"role": "user", "content": question})
    session["history"].append({"role": "assistant", "content": answer_text})
    session["turn_index"] = result.get("turn_index", session["turn_index"] + 1)
    session["case_memory"] = result.get("case_memory", session.get("case_memory", {}))
    return {
        "session_id": session["session_id"],
        "response_type": result.get("response_type", "answer"),
        "answer": answer_text,
        "treatment": result.get("treatment", ""),
        "matched_case": result.get("matched_case", {}),
        "candidate_count": result.get("candidate_count", 0),
        "top_candidates": result.get("top_candidates", []),
        "selected_question": result.get("selected_question", {}),
        "memory": session["case_memory"],
        "sources": result.get("sources", []),
        "confidence": result.get("confidence", 0.0),
        "phase": result.get("phase", ""),
        "turn_index": session["turn_index"],
        "best_score": result.get("best_score", 0.0),
        "intent": result.get("intent", ""),
        "query_en": result.get("query_en", ""),
    }


def run_cli(debug: bool = False) -> None:
    session = _get_or_create_session(None)
    sid = session["session_id"]
    print(f"Medical Assistant CLI（session={sid}）")
    print("输入 exit 退出\n")
    while True:
        try:
            q = input("[你] ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n退出。")
            break
        if not q or q.lower() == "exit":
            break
        try:
            result = run_one_turn(sid, q)
        except Exception as exc:
            print(f"\n[错误] {exc}\n")
            continue
        print(f"\n[助手] {result['answer']}\n")
        if debug:
            print(f"  type={result['response_type']} confidence={result['confidence']} "
                  f"phase={result['phase']} candidates={result['candidate_count']}")
            print(f"  query_en={result['query_en']} intent={result['intent']} "
                  f"best_score={result['best_score']}")
            print(f"  top_candidates={result['top_candidates']}")
            print(f"  selected_question={result['selected_question']}")
            print(f"  memory={result['memory']}\n")


settings = get_settings()
app = FastAPI(title=settings.app_name)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str = Field(min_length=1)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/chat")
def chat(req: ChatRequest):
    try:
        return run_one_turn(req.session_id, req.message.strip())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", nargs="?", choices=["cli", "api"], default="cli")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.mode == "api":
        uvicorn.run(app, host=settings.api_host, port=settings.api_port)
    else:
        run_cli(debug=args.debug)


if __name__ == "__main__":
    main()
