"""病例数据加载与文本表示。只管数据，不管评分。"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Sequence

from medical_assistant.config import get_settings
from medical_assistant.schemas import CaseRecord, ScoredCase
from medical_assistant.text.split import clean


def _list_field(item: dict, *names: str) -> list[str]:
    """从 JSON item 中提取字符串列表字段，去重。"""
    out = []
    for name in names:
        val = item.get(name)
        if isinstance(val, str) and val.strip():
            out.append(val.strip())
        elif isinstance(val, list):
            out.extend(str(x).strip() for x in val if str(x).strip())
    seen = set()
    return [v for v in out if v not in seen and not seen.add(v)]  # type: ignore


def extra_texts(case: CaseRecord | ScoredCase) -> list[str]:
    """收集病例的可选多语言/搜索字段。"""
    texts = []
    for attr in ("title_en", "description_en", "aliases", "keywords",
                 "key_terms_en", "search_terms"):
        val = getattr(case, attr, None)
        if isinstance(val, str) and val.strip():
            texts.append(val.strip())
        elif isinstance(val, (list, tuple, set)):
            texts.extend(str(x).strip() for x in val if str(x).strip())
    return texts


def full_text(case: CaseRecord | ScoredCase) -> str:
    """用于搜索/匹配的完整文本。"""
    parts = [case.title, case.description, case.treat]
    parts.extend(case.feature_tags or [])
    parts.extend(extra_texts(case))
    return "\n".join(clean(str(p)) for p in parts if clean(str(p)))


def document_text(case: CaseRecord) -> str:
    """用于向量索引的结构化文档文本。"""
    parts = [f"标题: {case.title}", f"描述: {case.description}", f"处理: {case.treat}"]
    if case.title_en:
        parts.append(f"Title: {case.title_en}")
    tags = [*case.aliases, *case.keywords, *case.key_terms_en,
            *case.search_terms, *case.feature_tags]
    if tags:
        parts.append("Tags: " + " / ".join(tags))
    return "\n".join(p for p in parts if p.strip())


@lru_cache(maxsize=1)
def load_cases() -> list[CaseRecord]:
    """加载全部病例。"""
    path = get_settings().case_data_path
    if not path.exists():
        raise FileNotFoundError(f"Case data not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases = []
    for i, item in enumerate(raw, 1):
        if not isinstance(item, dict):
            continue
        cases.append(CaseRecord(
            case_id=str(item.get("case_id") or f"case_{i:03d}"),
            title=str(item.get("title", "")).strip(),
            description=str(item.get("description", "")).strip(),
            treat=str(item.get("treat") or item.get("treatment") or "").strip(),
            title_en=str(item.get("title_en") or "").strip(),
            description_en=str(item.get("description_en") or "").strip(),
            aliases=_list_field(item, "aliases", "synonyms"),
            keywords=_list_field(item, "keywords"),
            key_terms_en=_list_field(item, "key_terms_en"),
            search_terms=_list_field(item, "search_terms"),
            feature_tags=_list_field(item, "feature_tags"),
        ))
    return cases


@lru_cache(maxsize=1)
def _case_map() -> dict[str, CaseRecord]:
    return {c.case_id: c for c in load_cases()}


def get_case(case_id: str) -> CaseRecord | None:
    return _case_map().get(case_id)


def get_cases(ids: Sequence[str] | None = None) -> list[CaseRecord]:
    if ids is None:
        return load_cases()
    m = _case_map()
    return [m[cid] for cid in ids if cid in m]


def all_ids() -> list[str]:
    return [c.case_id for c in load_cases()]
