"""启动入口：CLI / API。"""
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

_wf = None


def _workflow():
    global _wf
    if _wf is None:
        _wf = build_workflow()
    return _wf


# ── 会话管理 ──

_sessions: dict[str, dict] = {}


def _session(sid: str | None) -> dict:
    if sid and sid in _sessions:
        return _sessions[sid]
    s = {"id": sid or str(uuid4()), "history": [], "turn": 0,
         "memory": {}, "created": time.time()}
    _sessions[s["id"]] = s
    return s


# ── 单轮对话 ──

def chat(sid: str | None, user_input: str) -> dict[str, Any]:
    sess = _session(sid)
    result = _workflow().invoke({
        "user_input": user_input,
        "turn": sess["turn"],
        "history": sess["history"],
        "memory": sess["memory"],
    })

    reply = result.get("reply", "")
    sess["history"].append({"role": "user", "content": user_input})
    sess["history"].append({"role": "assistant", "content": reply})
    sess["turn"] = result.get("turn", sess["turn"] + 1)
    sess["memory"] = result.get("memory", sess["memory"])

    return {
        "session_id": sess["id"],
        "reply": reply,
        "reply_type": result.get("reply_type", "answer"),
        "probe": result.get("probe", {}),
        "matched_case": result.get("matched_case", {}),
        "confidence": result.get("confidence", 0.0),
        "candidate_count": result.get("candidate_count", 0),
        "top_candidates": result.get("top_candidates", []),
        "best_score": result.get("best_score", 0.0),
        "turn": sess["turn"],
        "memory": sess["memory"],
    }


# ── CLI ──

def cli(debug: bool = False) -> None:
    sess = _session(None)
    print(f"Medical Assistant CLI（session={sess['id']}）")
    print("输入 exit 退出\n")
    while True:
        try:
            q = input("[你] ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n退出。"); break
        if not q or q.lower() == "exit":
            break
        try:
            r = chat(sess["id"], q)
        except Exception as e:
            print(f"\n[错误] {e}\n"); continue
        print(f"\n[助手] {r['reply']}\n")
        if debug:
            print(f"  type={r['reply_type']} conf={r['confidence']} "
                  f"candidates={r['candidate_count']} best={r['best_score']}")
            print(f"  top={r['top_candidates']}")
            print(f"  probe={r['probe']}")
            print(f"  memory={r['memory']}\n")


# ── API ──

cfg = get_settings()
app = FastAPI(title=cfg.app_name)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])


class ChatReq(BaseModel):
    session_id: str | None = None
    message: str = Field(min_length=1)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/chat")
def api_chat(req: ChatReq):
    try:
        return chat(req.session_id, req.message.strip())
    except Exception as e:
        raise HTTPException(500, str(e)) from e


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", nargs="?", choices=["cli", "api"], default="cli")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()
    if args.mode == "api":
        uvicorn.run(app, host=cfg.api_host, port=cfg.api_port)
    else:
        cli(debug=args.debug)


if __name__ == "__main__":
    main()
