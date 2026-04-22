from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from medical_assistant.schemas.confidence import ConfidenceState
from medical_assistant.schemas.input import ConceptCandidate, Facet, NormalizedInput


def _merge_unique_strs(old: list[str], new: list[str], limit: int | None = None) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in [*(old or []), *(new or [])]:
        text = (item or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
        if limit and len(result) >= limit:
            break
    return result


def _merge_facets(old: list[Facet], new: list[Facet]) -> list[Facet]:
    bucket: dict[str, Facet] = {}
    for item in old or []:
        bucket[item.name] = item
    for item in new or []:
        previous = bucket.get(item.name)
        if previous is None:
            bucket[item.name] = item
            continue
        if item.turn > previous.turn or item.confidence >= previous.confidence:
            bucket[item.name] = item
    return list(bucket.values())


def _merge_candidates(old: list[ConceptCandidate], new: list[ConceptCandidate]) -> list[ConceptCandidate]:
    bucket: dict[str, ConceptCandidate] = {}
    for item in [*(old or []), *(new or [])]:
        key = (item.term or "").strip().lower()
        if not key:
            continue
        previous = bucket.get(key)
        if previous is None or item.support > previous.support:
            bucket[key] = item
    return sorted(bucket.values(), key=lambda x: x.support, reverse=True)[:5]


class MedicalContext(BaseModel):
    model_config = ConfigDict(extra="ignore")

    latest_query_en: str | None = None
    query_history_en: list[str] = Field(default_factory=list)
    latest_query_bundle: NormalizedInput | None = None
    keyword_history: list[str] = Field(default_factory=list)

    # Compatibility state kept for downstream nodes that still read them.
    chief_complaint: str | None = None
    candidate_concepts: list[ConceptCandidate] = Field(default_factory=list)
    normalized_terms: list[str] = Field(default_factory=list)
    facets: list[Facet] = Field(default_factory=list)
    negated_findings: list[Facet] = Field(default_factory=list)
    red_flags: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    free_text_observations: list[str] = Field(default_factory=list)

    def facet_names(self) -> set[str]:
        names = {item.name for item in self.facets}
        names.update(item.name for item in self.negated_findings)
        return names

    def merge_normalized(self, normalized: NormalizedInput) -> None:
        prepared = normalized.prepare_for_runtime()
        self.latest_query_bundle = prepared

        query_en = prepared.primary_query()
        if query_en:
            self.latest_query_en = query_en
            self.query_history_en = _merge_unique_strs(self.query_history_en, [query_en], limit=12)

        derived_terms = prepared.keyword_terms(limit=12) or prepared.normalized_terms
        self.keyword_history = _merge_unique_strs(self.keyword_history, derived_terms, limit=20)

        if prepared.chief_complaint:
            self.chief_complaint = prepared.chief_complaint

        self.normalized_terms = _merge_unique_strs(
            self.normalized_terms,
            derived_terms,
            limit=12,
        )
        self.red_flags = _merge_unique_strs(self.red_flags, prepared.red_flags, limit=8)
        self.unresolved_questions = _merge_unique_strs(
            prepared.unresolved_questions,
            self.unresolved_questions,
            limit=6,
        )
        self.free_text_observations = (
            self.free_text_observations + prepared.free_text_observations
        )[-5:]
        self.facets = _merge_facets(self.facets, prepared.facets)
        self.negated_findings = _merge_facets(self.negated_findings, prepared.negated_findings)
        self.candidate_concepts = _merge_candidates(
            self.candidate_concepts,
            prepared.candidate_concepts,
        )


class ConversationState(BaseModel):
    model_config = ConfigDict(extra="ignore")

    session_id: str | None = None
    turn_index: int = 0
    phase: Literal[
        "NEW",
        "NORMALIZED",
        "RETRIEVED",
        "NEEDS_CLARIFICATION",
        "READY_TO_ANSWER",
        "SAFETY_ESCALATION",
        "ANSWERED",
    ] = "NEW"
    raw_user_utterances: list[str] = Field(default_factory=list)
    medical_context: MedicalContext = Field(default_factory=MedicalContext)
    asked_questions: list[str] = Field(default_factory=list)
    last_clarify_question: str | None = None
    last_response_type: Literal["answer", "clarification", "safety"] | None = None
    confidence: ConfidenceState = Field(default_factory=ConfidenceState)

    def register_user_turn(self, text: str) -> None:
        self.turn_index += 1
        self.raw_user_utterances.append(text)

    def add_asked_question(self, question: str) -> None:
        self.asked_questions = _merge_unique_strs(self.asked_questions, [question], limit=20)

    def snapshot(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "turn_index": self.turn_index,
            "latest_query_en": self.medical_context.latest_query_en,
            "normalized_terms": self.medical_context.normalized_terms,
            "red_flags": self.medical_context.red_flags,
            "unresolved_questions": self.medical_context.unresolved_questions,
            "last_clarify_question": self.last_clarify_question,
            "confidence": self.confidence.model_dump(mode="json"),
        }
