"""Pydantic 数据模型。"""
from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class NormalizedInput(BaseModel):
    query_en: str = ""
    intent: str = "general"
    key_terms_en: list[str] = Field(default_factory=list)


class ProbeAnswerParse(BaseModel):
    signal: Literal["yes", "no", "uncertain", "unrelated"] = "unrelated"
    confidence: float = 0.0
    evidence: str = ""
    new_observations: list[str] = Field(default_factory=list)


class CaseRecord(BaseModel):
    case_id: str
    title: str
    description: str
    treat: str
    title_en: str = ""
    description_en: str = ""
    aliases: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    key_terms_en: list[str] = Field(default_factory=list)
    search_terms: list[str] = Field(default_factory=list)
    feature_tags: list[str] = Field(default_factory=list)


class CaseCandidate(CaseRecord):
    score: float = 0.0
    matched_features: list[str] = Field(default_factory=list)
    matched_terms: list[str] = Field(default_factory=list)


class PlannedQuestion(BaseModel):
    question_id: str
    feature_id: str
    text: str
    positive_case_ids: list[str] = Field(default_factory=list)
    negative_case_ids: list[str] = Field(default_factory=list)
    unknown_case_ids: list[str] = Field(default_factory=list)
    split_score: float = 0.0
    label: str = ""
    strategy: str = ""
    tree_node_id: str = ""
    yes_child_id: str = ""
    no_child_id: str = ""
    evidence_texts: list[str] = Field(default_factory=list)
    debug: dict[str, Any] = Field(default_factory=dict)


AnswerSignal = Literal["yes", "no", "uncertain", "unrelated"]


class CaseMemory(BaseModel):
    original_question: str = ""
    normalized_query: str = ""
    key_terms: list[str] = Field(default_factory=list)

    confirmed_features: dict[str, Any] = Field(default_factory=dict)
    denied_features: dict[str, Any] = Field(default_factory=dict)
    uncertain_features: dict[str, Any] = Field(default_factory=dict)

    candidate_case_ids: list[str] = Field(default_factory=list)
    asked_question_ids: list[str] = Field(default_factory=list)
    asked_feature_ids: list[str] = Field(default_factory=list)

    last_question_id: str = ""
    last_question_text: str = ""
    last_question_feature: str = ""
    pending_answer_feature: str = ""
    pending_answer_signal: AnswerSignal | str = ""

    tree_node_id: str = ""
    last_question_tree_node_id: str = ""
    last_question_yes_child: str = ""
    last_question_no_child: str = ""

    probe_splits: dict[str, dict[str, list[str]]] = Field(default_factory=dict)
    probe_labels: dict[str, str] = Field(default_factory=dict)
    probe_questions: dict[str, str] = Field(default_factory=dict)

    resolved_case_id: str = ""
    turn_index: int = 0
