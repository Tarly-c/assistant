"""Pydantic models for the case-localization demo."""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class NormalizedInput(BaseModel):
    """normalize 节点让 LLM 填的模型。"""

    query_en: str = Field(default="", description="用户问题翻译成英文的简洁检索查询")
    intent: str = Field(
        default="general",
        description="意图分类，只能是: treatment / cause / symptom / diagnosis / general",
    )
    key_terms_en: list[str] = Field(
        default_factory=list,
        description="最多 5 个关键医学术语（英文或数据集内关键词）",
    )


class AnswerDraft(BaseModel):
    answer: str = Field(default="", description="给用户的中文回答，200 字以内")
    sources_used: list[str] = Field(default_factory=list, description="回答中引用了哪些来源标题")


class ClarifyDraft(BaseModel):
    question: str = Field(default="", description="需要用户补充的追问（中文，1 个问题）")


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

    # Optional bilingual/search fields. Chinese-only JSON does not need them;
    # if future data includes them, tree mining and retrieval use them.
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
    """Cross-turn structured memory."""

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

    def to_public_dict(self) -> dict[str, Any]:
        return self.model_dump()
