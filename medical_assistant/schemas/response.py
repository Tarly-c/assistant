from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from medical_assistant.schemas.confidence import ConfidenceState
from medical_assistant.schemas.retrieval import CandidateTopic


class SafetyAssessment(BaseModel):
    model_config = ConfigDict(extra="ignore")

    level: Literal["normal", "high"] = "normal"
    reasons: list[str] = Field(default_factory=list)
    missing_checks: list[str] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    source_type: Literal["local", "pubmed"]
    label: str
    title: str
    detail: str = ""


class AssistantResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    response_type: Literal["answer", "clarification", "safety"]
    content: str
    matched_topics: list[CandidateTopic] = Field(default_factory=list)
    confidence: ConfidenceState | None = None
    safety: SafetyAssessment | None = None
    sources: list[EvidenceItem] = Field(default_factory=list)
    next_question: str | None = None
