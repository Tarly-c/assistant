from __future__ import annotations

from pydantic import BaseModel, Field
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from medical_assistant.config import get_settings
from medical_assistant.graph.workflow import build_workflow
from medical_assistant.schemas.response import AssistantResponse
from medical_assistant.schemas.state import ConversationState
from medical_assistant.storage.session_store import InMemorySessionStore
from medical_assistant.storage.trace_store import JsonlTraceStore


settings = get_settings()
workflow = build_workflow()
session_store = InMemorySessionStore()
trace_store = JsonlTraceStore()

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
    state: dict


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str = Field(min_length=1)


@web_app.post("/api/session", response_model=SessionCreateResponse)
def create_session() -> SessionCreateResponse:
    record = session_store.create()
    return SessionCreateResponse(
        session_id=record.session_id,
        state=record.state.snapshot(),
    )


@web_app.post("/api/chat")
def chat(payload: ChatRequest) -> dict:
    record = session_store.get_or_create(payload.session_id)
    question = payload.message.strip()

    result = workflow.invoke(
        {
            "question": question,
            "conversation_state": record.state.model_dump(mode="json"),
        }
    )

    updated_state = ConversationState.model_validate(
        result.get("conversation_state") or record.state.model_dump(mode="json")
    )
    response = AssistantResponse.model_validate(
        result.get("response") or {"response_type": "answer", "content": ""}
    )

    session_store.save(record.session_id, updated_state)
    session_store.append_message(
        record.session_id,
        {
            "user": question,
            "assistant": response.content,
            "response_type": response.response_type,
            "state": updated_state.snapshot(),
            "confidence": updated_state.confidence.model_dump(mode="json"),
        },
    )

    trace_store.write(
        record.session_id,
        {
            "question": question,
            "plan": result.get("plan", {}),
            "local_result": result.get("local_result", {}),
            "candidate_topics": result.get("candidate_topics", []),
            "safety": result.get("safety", {}),
            "confidence": result.get("confidence", {}),
            "web_result": result.get("web_result", {}),
            "response": response.model_dump(mode="json"),
            "state": updated_state.model_dump(mode="json"),
        },
    )

    return {
        "session_id": record.session_id,
        "response_type": response.response_type,
        "answer": response.content,
        "matched_topics": [item.model_dump(mode="json") for item in response.matched_topics],
        "confidence": updated_state.confidence.model_dump(mode="json"),
        "safety": response.safety.model_dump(mode="json") if response.safety else {},
        "plan": result.get("plan", {}),
        "local_result": result.get("local_result", {}),
        "pubmed_result": result.get("web_result", {}),
        "state": updated_state.snapshot(),
        "history_count": len(session_store.get(record.session_id).messages),
    }
