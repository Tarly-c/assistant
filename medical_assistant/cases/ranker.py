"""病例评分、排序、过滤。"""
from __future__ import annotations

from typing import Iterable

from medical_assistant.config import get_settings
from medical_assistant.schemas import CaseRecord, ScoredCase
from medical_assistant.text.split import extract_terms, norm
from medical_assistant.cases.store import all_ids, document_text, get_cases


def _hits(haystack: str, terms: Iterable[str]) -> list[str]:
    """返回 terms 中能在 haystack 里子串命中的项。"""
    h = norm(haystack)
    return [t for t in terms if norm(t) in h]


def _membership(case_id: str, probe_id: str,
                splits: dict[str, dict[str, list[str]]]) -> str:
    """查询 case_id 在某个 probe 切分中的归属。"""
    s = splits.get(probe_id, {})
    if case_id in set(s.get("positive", [])):
        return "positive"
    if case_id in set(s.get("negative", [])):
        return "negative"
    if case_id in set(s.get("unknown", [])):
        return "unknown"
    return "missing"


def score_case(
    case: CaseRecord, query: str, terms: list[str],
    confirmed: Iterable[str], denied: Iterable[str],
    splits: dict[str, dict[str, list[str]]],
) -> tuple[float, list[str], list[str]]:
    """为单个病例计算匹配分数。返回 (score, hit_probes, hit_terms)。"""
    text = document_text(case)
    title_text = "\n".join([case.title, case.title_en, *case.aliases])

    # 搜索词（query + terms + query 的子片段）
    search = []
    q = norm(query)
    if q and len(q) > 1:
        search.append(query)
    for t in (terms or []):
        if t not in search:
            search.append(t)
    for t in extract_terms(query):
        if t not in search:
            search.append(t)

    hit_terms = _hits(text, search)
    title_hits = _hits(title_text, search)
    s = 0.0
    if search:
        s += 0.35 * len(hit_terms) / max(1, len(search))
        s += 0.15 * len(title_hits) / max(1, len(search))

    # confirmed probe 奖惩
    hit_probes = []
    for pid in (confirmed or []):
        m = _membership(case.case_id, pid, splits)
        if m == "positive":
            s += 0.24; hit_probes.append(pid)
        elif m == "negative":
            s -= 0.35
        elif m == "unknown":
            s += 0.03

    # denied probe 奖惩
    for pid in (denied or []):
        m = _membership(case.case_id, pid, splits)
        if m == "negative":
            s += 0.18
        elif m == "positive":
            s -= 0.32
        elif m == "unknown":
            s += 0.02

    return round(max(0.0, min(1.0, s)), 4), hit_probes, hit_terms


def rank_cases(
    query: str, terms: list[str],
    candidate_ids: list[str] | None = None,
    confirmed: Iterable[str] = (), denied: Iterable[str] = (),
    splits: dict[str, dict[str, list[str]]] | None = None,
) -> list[ScoredCase]:
    """对候选病例评分并排序。"""
    splits = splits or {}
    ranked = []
    for case in get_cases(candidate_ids):
        s, hp, ht = score_case(case, query, terms or [], confirmed, denied, splits)
        ranked.append(ScoredCase(**case.model_dump(), score=s, hit_probes=hp, hit_terms=ht))
    ranked.sort(key=lambda c: c.score, reverse=True)
    return ranked


def filter_by_probes(
    ids: list[str] | None,
    confirmed: Iterable[str], denied: Iterable[str],
    splits: dict[str, dict[str, list[str]]],
) -> list[str]:
    """根据 confirmed/denied probe 过滤病例 ID 列表。"""
    result = list(ids or all_ids())
    for pid in (confirmed or []):
        s = splits.get(pid)
        if not s:
            continue
        allowed = set(s.get("positive", []) + s.get("unknown", []))
        narrowed = [cid for cid in result if cid in allowed]
        if narrowed:
            result = narrowed
    for pid in (denied or []):
        s = splits.get(pid)
        if not s:
            continue
        allowed = set(s.get("negative", []) + s.get("unknown", []))
        narrowed = [cid for cid in result if cid in allowed]
        if narrowed:
            result = narrowed
    return result


def confidence(candidates: list[ScoredCase]) -> float:
    """从候选列表计算置信度。"""
    if not candidates:
        return 0.0
    if len(candidates) == 1:
        return max(0.65, candidates[0].score)
    gap = max(0.0, candidates[0].score - candidates[1].score)
    return round(min(1.0, max(candidates[0].score, 0.5 + gap)), 4)
