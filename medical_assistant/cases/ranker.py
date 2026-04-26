"""病例评分、排序、过滤。"""
from __future__ import annotations

from typing import Iterable

from medical_assistant.config import get_settings
from medical_assistant.schemas import CaseCandidate, CaseRecord
from medical_assistant.text.split import extract_search_terms, norm_text
from medical_assistant.cases.store import (
    all_case_ids, case_document_text, get_cases, case_extra_texts,
)


def _term_hits(text: str, terms: Iterable[str]) -> list[str]:
    haystack = norm_text(text)
    return [t for t in terms if norm_text(t) in haystack]


def _split_membership(
    case_id: str, probe_id: str,
    probe_splits: dict[str, dict[str, list[str]]] | None,
) -> str:
    split = (probe_splits or {}).get(probe_id, {})
    if case_id in set(split.get("positive", [])):
        return "positive"
    if case_id in set(split.get("negative", [])):
        return "negative"
    if case_id in set(split.get("unknown", [])):
        return "unknown"
    return "missing"


def _score_case(
    case: CaseRecord, query: str, terms: list[str],
    confirmed: Iterable[str], denied: Iterable[str],
    probe_splits: dict[str, dict[str, list[str]]] | None = None,
) -> tuple[float, list[str], list[str]]:
    text = case_document_text(case)
    title_text = "\n".join([case.title, case.title_en, *case.aliases, *case.key_terms_en])

    search_terms: list[str] = []
    q = norm_text(query)
    if q and len(q) > 1:
        search_terms.append(query)
    for t in (terms or []):
        if t not in search_terms:
            search_terms.append(t)
    for t in extract_search_terms(query):
        if t not in search_terms:
            search_terms.append(t)

    matched_terms = _term_hits(text, search_terms)
    title_hits = _term_hits(title_text, search_terms)
    score = 0.0
    if search_terms:
        score += 0.35 * (len(matched_terms) / max(1, len(search_terms)))
        score += 0.15 * (len(title_hits) / max(1, len(search_terms)))

    matched_features: list[str] = []
    for pid in (confirmed or []):
        m = _split_membership(case.case_id, pid, probe_splits)
        if m == "positive":
            score += 0.24
            matched_features.append(pid)
        elif m == "negative":
            score -= 0.35
        elif m == "unknown":
            score += 0.03
    for pid in (denied or []):
        m = _split_membership(case.case_id, pid, probe_splits)
        if m == "negative":
            score += 0.18
        elif m == "positive":
            score -= 0.32
        elif m == "unknown":
            score += 0.02

    return round(max(0.0, min(1.0, score)), 4), matched_features, matched_terms


def search_cases(
    query: str, terms: list[str] | None = None,
    candidate_ids: list[str] | None = None,
    confirmed_features: Iterable[str] | None = None,
    denied_features: Iterable[str] | None = None,
    probe_splits: dict[str, dict[str, list[str]]] | None = None,
    top_k: int | None = None,
) -> list[CaseCandidate]:
    settings = get_settings()
    records = get_cases(candidate_ids)
    ranked: list[CaseCandidate] = []
    for case in records:
        score, mf, mt = _score_case(
            case, query, terms or [],
            confirmed_features or [], denied_features or [],
            probe_splits,
        )
        ranked.append(CaseCandidate(**case.model_dump(), score=score,
                                    matched_features=mf, matched_terms=mt))
    ranked.sort(key=lambda c: c.score, reverse=True)
    limit = top_k if top_k is not None else settings.case_initial_top_k
    return ranked[:limit] if limit and limit > 0 else ranked


def apply_feature_filters(
    candidate_ids: list[str] | None,
    confirmed_features: Iterable[str],
    denied_features: Iterable[str],
    probe_splits: dict[str, dict[str, list[str]]] | None = None,
) -> list[str]:
    ids = list(candidate_ids or all_case_ids())
    splits = probe_splits or {}
    for pid in (confirmed_features or []):
        split = splits.get(pid)
        if not split:
            continue
        allowed = set(split.get("positive", []) + split.get("unknown", []))
        narrowed = [cid for cid in ids if cid in allowed]
        if narrowed:
            ids = narrowed
    for pid in (denied_features or []):
        split = splits.get(pid)
        if not split:
            continue
        allowed = set(split.get("negative", []) + split.get("unknown", []))
        narrowed = [cid for cid in ids if cid in allowed]
        if narrowed:
            ids = narrowed
    return ids


def confidence_from_candidates(candidates: list[CaseCandidate]) -> float:
    if not candidates:
        return 0.0
    if len(candidates) == 1:
        return max(0.65, candidates[0].score)
    gap = max(0.0, candidates[0].score - candidates[1].score)
    return round(min(1.0, max(candidates[0].score, 0.5 + gap)), 4)


def top_candidate_summary(candidates: list[CaseCandidate], limit: int | None = None) -> list[dict]:
    limit = limit or get_settings().case_display_top_k
    return [
        {"case_id": c.case_id, "title": c.title, "score": c.score,
         "matched_features": c.matched_features, "matched_terms": c.matched_terms}
        for c in candidates[:limit]
    ]
