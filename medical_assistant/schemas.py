"""所有 Pydantic 模型，集中在一个文件。"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ── LLM 结构化输出用 ──────────────────────────────────────
class NormalizedInput(BaseModel):
    """normalize 节点让 LLM 填的模型。

    demo 默认不强依赖 LLM；LLM 不可用时 normalize.py 会用规则兜底。
    """

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


# ── 病例定位 workflow 用 ───────────────────────────────────
class CaseRecord(BaseModel):
    case_id: str
    title: str
    description: str
    treat: str
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


AnswerSignal = Literal["yes", "no", "uncertain", "unrelated"]


class CaseMemory(BaseModel):
    """跨轮保留的结构化记忆。

    history 仍可用于 UI 展示，但病例定位不再依赖“最近两轮对话拼接”。
    """

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

    resolved_case_id: str = ""
    turn_index: int = 0

    def to_public_dict(self) -> dict[str, Any]:
        return self.model_dump()
