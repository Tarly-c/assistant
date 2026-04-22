from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


class RetrievalPlan(BaseModel):
    model_config = ConfigDict(extra="ignore")
    local_query_en: str = ""
    pubmed_queries: list[str] = Field(default_factory=list)
    intent: Literal["treatment", "cause", "symptom", "diagnosis", "general"] = "general"
    normalized_terms: list[str] = Field(default_factory=list)
    use_pubmed: bool = False


class RetrievalHit(BaseModel):
    model_config = ConfigDict(extra="ignore")
    source: str = ""
    title: str = ""
    chunk_id: str | None = None
    snippet: str = ""
    score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class LocalSearchResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    enough: bool = False
    score: float = 0.0
    reason: str = ""
    hits: list[RetrievalHit] = Field(default_factory=list)


class CandidateTopic(BaseModel):
    model_config = ConfigDict(extra="ignore")
    title: str
    source: str
    score: float = 0.0
    matched_terms: list[str] = Field(default_factory=list)
    missing_terms: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)


class PubMedHit(BaseModel):
    model_config = ConfigDict(extra="ignore")
    pmid: str = ""
    title: str = ""
    journal: str = ""
    pubdate: str = ""
    snippet: str = ""
    publication_types: list[str] = Field(default_factory=list)
    mesh_headings: list[str] = Field(default_factory=list)
    rerank_score: float = 0.0


class PubMedSearchResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    query: str = ""
    enough: bool = False
    score: float = 0.0
    reason: str = ""
    hits: list[PubMedHit] = Field(default_factory=list)
