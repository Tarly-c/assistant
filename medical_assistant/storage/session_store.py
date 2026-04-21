from __future__ import annotations

import threading
import time
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from medical_assistant.schemas.state import ConversationState


class SessionRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    session_id: str
    state: ConversationState
    messages: list[dict[str, Any]] = Field(default_factory=list)
    created_at: float
    updated_at: float
    active: bool = True


class InMemorySessionStore:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._sessions: dict[str, SessionRecord] = {}

    def create(self) -> SessionRecord:
        with self._lock:
            session_id = str(uuid4())
            now = time.time()
            record = SessionRecord(
                session_id=session_id,
                state=ConversationState(session_id=session_id),
                created_at=now,
                updated_at=now,
            )
            self._sessions[session_id] = record
            return record

    def get(self, session_id: str) -> SessionRecord:
        with self._lock:
            return self._sessions[session_id]

    def get_or_create(self, session_id: str | None = None) -> SessionRecord:
        with self._lock:
            if session_id and session_id in self._sessions:
                return self._sessions[session_id]
            return self.create()

    def save(self, session_id: str, state: ConversationState) -> SessionRecord:
        with self._lock:
            record = self._sessions[session_id]
            record.state = state
            record.updated_at = time.time()
            self._sessions[session_id] = record
            return record

    def append_message(self, session_id: str, payload: dict[str, Any]) -> SessionRecord:
        with self._lock:
            record = self._sessions[session_id]
            record.messages.append(payload)
            record.updated_at = time.time()
            self._sessions[session_id] = record
            return record
