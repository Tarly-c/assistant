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


class QueryKeyword(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str
    category: Literal[
        "symptom",
        "body_part",
        "time",
        "severity",
        "trigger",
        "negation",
        "modifier",
        "user_term",
        "unknown",
    ] = "unknown"
    normalized_en: str | None = None
    confidence: float = 0.0
    source: Literal["llm", "dictionary", "rule", "system"] = "llm"


class SearchQuery(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str
    channel: Literal["local", "web", "hybrid"] = "local"
    purpose: Literal["primary", "recall", "precision", "expansion"] = "primary"
    weight: float = 1.0


class NormalizedInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    original_text: str = ""
    translated_en: str = ""
    keywords: list[QueryKeyword] = Field(default_factory=list)
    queries: list[SearchQuery] = Field(default_factory=list)
    follow_up_hints: list[str] = Field(default_factory=list)

    # Legacy compatibility fields. These remain optional shells so downstream
    # nodes can keep running while the pipeline migrates to query bundles.
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

    def prepare_for_runtime(self) -> "NormalizedInput":
        if not self.original_text:
            primary_query = next((item.text for item in self.queries if item.text.strip()), "")
            self.original_text = primary_query or self.translated_en

        if not self.translated_en:
            self.translated_en = self.original_text

        if not self.queries:
            self.queries = [
                SearchQuery(
                    text=self.translated_en or self.original_text,
                    channel="local",
                    purpose="primary",
                    weight=1.0,
                )
            ]

        if not self.normalized_terms:
            self.normalized_terms = self.keyword_terms(limit=8)

        if not self.search_keywords:
            self.search_keywords = list(self.normalized_terms)

        if not self.chief_complaint:
            self.chief_complaint = self.primary_keyword(category_priority=("symptom", "user_term"))

        if not self.unresolved_questions and self.follow_up_hints:
            self.unresolved_questions = list(dict.fromkeys(self.follow_up_hints))[:4]

        return self

    def keyword_terms(self, limit: int | None = None) -> list[str]:
        values: list[str] = []
        seen: set[str] = set()
        for item in self.keywords:
            text = (item.normalized_en or item.text or "").strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            values.append(text)
            if limit is not None and len(values) >= limit:
                break
        return values

    def primary_query(self) -> str:
        for item in self.queries:
            text = (item.text or "").strip()
            if text:
                return text
        return (self.translated_en or self.original_text or "").strip()

    def primary_keyword(
        self,
        category_priority: tuple[str, ...] = ("symptom", "body_part", "user_term"),
    ) -> str | None:
        for category in category_priority:
            for item in self.keywords:
                if item.category == category:
                    text = (item.normalized_en or item.text or "").strip()
                    if text:
                        return text
        return None
