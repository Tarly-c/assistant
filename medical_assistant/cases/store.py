"""病例数据加载、查找、文本表示。只管数据，不管评分。"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Sequence

from medical_assistant.config import get_settings
from medical_assistant.schemas import CaseCandidate, CaseRecord
from medical_assistant.text.split import readable


def _list_field(item: dict, *names: str) -> list[str]:
    out: list[str] = []
    for name in names:
        val = item.get(name)
        if isinstance(val, str) and val.strip():
            out.append(val.strip())
        elif isinstance(val, list):
            out.extend(str(x).strip() for x in val if str(x).strip())
    seen: set[str] = set()
    return [v for v in out if v not in seen and not seen.add(v)]  # type: ignore[func-returns-value]


def case_extra_texts(case: CaseRecord | CaseCandidate) -> list[str]:
    """收集可选的多语言/搜索字段。"""
    texts: list[str] = []
    for attr in ("title_en", "description_en", "aliases", "keywords",
                 "key_terms_en", "search_terms"):
        val = getattr(case, attr, None)
        if isinstance(val, str) and val.strip():
            texts.append(val.strip())
        elif isinstance(val, (list, tuple, set)):
            texts.extend(str(x).strip() for x in val if str(x).strip())
    return texts


def case_search_text(case: CaseRecord | CaseCandidate) -> str:
    """用于搜索/匹配的完整文本。"""
    parts = [case.title, case.description, case.treat]
    parts.extend(case.feature_tags or [])
    parts.extend(case_extra_texts(case))
    return "\n".join(readable(str(p)) for p in parts if readable(str(p)))


def case_document_text(case: CaseRecord) -> str:
    """用于向量索引的结构化文档文本。"""
    parts = [f"标题: {case.title}", f"描述: {case.description}", f"处理: {case.treat}"]
    if case.title_en:
        parts.append(f"Title EN: {case.title_en}")
    if case.description_en:
        parts.append(f"Description EN: {case.description_en}")
    extra = [*case.aliases, *case.keywords, *case.key_terms_en,
             *case.search_terms, *case.feature_tags]
    if extra:
        parts.append("Search terms: " + " / ".join(extra))
    return "\n".join(p for p in parts if p.strip())


@lru_cache(maxsize=1)
def load_cases() -> list[CaseRecord]:
    path = get_settings().case_data_path
    if not path.exists():
        raise FileNotFoundError(f"Case data not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Case data must be a JSON list")
    cases: list[CaseRecord] = []
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
            aliases=_list_field(item, "aliases", "alias", "synonyms"),
            keywords=_list_field(item, "keywords", "keyword"),
            key_terms_en=_list_field(item, "key_terms_en", "terms_en"),
            search_terms=_list_field(item, "search_terms", "search_aliases"),
            feature_tags=_list_field(item, "feature_tags"),
        ))
    return cases


@lru_cache(maxsize=1)
def case_map() -> dict[str, CaseRecord]:
    return {c.case_id: c for c in load_cases()}


def clear_case_cache() -> None:
    load_cases.cache_clear()
    case_map.cache_clear()


def get_case(case_id: str) -> CaseRecord | None:
    return case_map().get(case_id)


def get_cases(case_ids: Sequence[str] | None = None) -> list[CaseRecord]:
    if case_ids is None:
        return load_cases()
    m = case_map()
    return [m[cid] for cid in case_ids if cid in m]


def all_case_ids() -> list[str]:
    return [c.case_id for c in load_cases()]
