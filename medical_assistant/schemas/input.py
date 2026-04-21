from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Facet(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    value: str | bool | int | float | None = None
    status: Literal["present", "negated", "uncertain", "unknown"] = "present"
    source: Literal["user", "llm", "dictionary", "system"] = "user"
    confidence: float = 1.0
    turn: int = 1


class ConceptCandidate(BaseModel):
    model_config = ConfigDict(extra="ignore")

    term: str
    concept_id: str | None = None
    support: float = 0.0
    source: Literal["llm", "dictionary", "retrieval"] = "llm"


class NormalizedInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    original_text: str = ""
    chief_complaint: str | None = None
    normalized_terms: list[str] = Field(default_factory=list)
    search_keywords: list[str] = Field(default_factory=list)
    candidate_concepts: list[ConceptCandidate] = Field(default_factory=list)
    facets: list[Facet] = Field(default_factory=list)
    negated_findings: list[Facet] = Field(default_factory=list)
    red_flags: list[str] = Field(default_factory=list)
    uncertain_fields: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    free_text_observations: list[str] = Field(default_factory=list)
