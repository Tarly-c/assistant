"""所有 Pydantic 模型，集中在一个文件。"""
from __future__ import annotations
from pydantic import BaseModel, Field


# ── LLM 结构化输出用（字段越少越好） ──────────────────────

class NormalizedInput(BaseModel):
    """normalize 节点让 LLM 填的唯一模型：3 个字段。"""
    query_en: str = Field(
        default="",
        description="用户问题翻译成英文的简洁检索查询",
    )
    intent: str = Field(
        default="general",
        description="意图分类，只能是: treatment / cause / symptom / diagnosis / general",
    )
    key_terms_en: list[str] = Field(
        default_factory=list,
        description="最多 5 个关键医学术语（英文）",
    )


class AnswerDraft(BaseModel):
    """answer 节点让 LLM 填的模型：2 个字段。"""
    answer: str = Field(
        default="",
        description="给用户的中文回答，200 字以内",
    )
    sources_used: list[str] = Field(
        default_factory=list,
        description="回答中引用了哪些来源标题",
    )


class ClarifyDraft(BaseModel):
    """clarify 节点让 LLM 填的模型：1 个字段。"""
    question: str = Field(
        default="",
        description="需要用户补充的追问（中文，1 个问题）",
    )
