from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ConfidenceState(BaseModel):
    model_config = ConfigDict(extra="ignore")

    mapping_confidence: float = 0.0
    retrieval_confidence: float = 0.0
    dialog_confidence: float = 0.0
    safety_confidence: float = 0.0
    overall_confidence: float = 0.0
    reasons: list[str] = Field(default_factory=list)
