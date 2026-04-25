from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from medical_assistant.config import get_settings
from medical_assistant.schemas import CaseCandidate, CaseRecord
from medical_assistant.services.cases.features import extract_features


GENERIC_QUERY_WORDS = {"牙疼", "牙痛", "疼", "痛", "toothache", "tooth pain", "teeth pain"}


def make_case_id(index: int) -> str:
    return f"case_{index:03d}"


def case_document_text(case: CaseRecord) -> str:
    tags = "、".join(case.feature_tags)
    return f"标题: {case.title}\n描述: {case.description}\n特征: {tags}\n处理: {case.treat}"


@lru_cache(maxsize=1)
def load_cases() -> list[CaseRecord]:
    settings = get_settings()
    path = settings.case_data_path
    if not path.exists():
        raise FileNotFoundError(
            f"Case data not found: {path}. Put the JSON file at this path or set "
            "MEDICAL_ASSISTANT_CASE_DATA_FILE."
        )

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Case data must be a JSON list")

    cases: list[CaseRecord] = []
    for i, item in enumerate(raw, 1):
        title = str(item.get("title", "")).strip()
        description = str(item.get("description", "")).strip()
        treat = str(item.get("treat", "")).strip()
        feature_tags = extract_features(title)
        cases.append(
            CaseRecord(
                case_id=make_case_id(i),
                title=title,
                description=description,
                treat=treat,
                feature_tags=feature_tags,
            )
        )
    return cases


@lru_cache(maxsize=1)
def case_map() -> dict[str, CaseRecord]:
    return {c.case_id: c for c in load_cases()}


def get_case(case_id: str) -> CaseRecord | None:
    return case_map().get(case_id)


def get_cases(case_ids: Iterable[str] | None = None) -> list[CaseRecord]:
    if case_ids is None:
        return load_cases()
    cmap = case_map()
    return [cmap[cid] for cid in case_ids if cid in cmap]


def all_case_ids() -> list[str]:
    return [c.case_id for c in load_cases()]


def _norm(text: str) -> str:
    return (text or "").lower().strip()


def _term_hits(text: str, terms: Iterable[str]) -> list[str]:
    haystack = _norm(text)
    hits: list[str] = []
    for term in terms:
        t = _norm(term)
        if t and t in haystack and t not in hits:
            hits.append(term)
    return hits


def _query_is_generic(query: str, terms: list[str]) -> bool:
    q = _norm(query)
    return (not q or q in GENERIC_QUERY_WORDS or len(q) <= 3) and not terms


def _score_case(
    case: CaseRecord,
    query: str,
    terms: list[str],
    confirmed_features: Iterable[str],
    denied_features: Iterable[str],
) -> tuple[float, list[str], list[str]]:
    text = case_document_text(case)
    title_text = case.title
    confirmed = set(confirmed_features or [])
    denied = set(denied_features or [])
    tags = set(case.feature_tags)

    query_features = set(extract_features(query))
    all_positive_features = confirmed | query_features

    matched_features = sorted(tags & all_positive_features)
    denied_conflicts = sorted(tags & denied)

    search_terms = [query] if query and not _query_is_generic(query, terms) else []
    search_terms += terms
    matched_terms = _term_hits(text, search_terms)
    title_hits = _term_hits(title_text, search_terms)

    score = 0.0
    if all_positive_features:
        score += 0.55 * (len(tags & all_positive_features) / max(1, len(all_positive_features)))
    if denied:
        score -= 0.35 * (len(tags & denied) / max(1, len(denied)))
    if search_terms:
        score += 0.25 * (len(matched_terms) / max(1, len(search_terms)))
        score += 0.10 * (len(title_hits) / max(1, len(search_terms)))
    if query_features and tags & query_features:
        score += 0.10
    if denied_conflicts:
        score -= 0.05 * len(denied_conflicts)

    # Keep score in a predictable range for API display.
    score = max(0.0, min(1.0, score))
    return round(score, 4), matched_features, matched_terms


def search_cases(
    query: str,
    terms: list[str] | None = None,
    candidate_ids: list[str] | None = None,
    confirmed_features: Iterable[str] | None = None,
    denied_features: Iterable[str] | None = None,
    top_k: int | None = None,
) -> list[CaseCandidate]:
    """Rank case-level records.

    This function intentionally returns cases, not document chunks. The Chroma
    index built by scripts/build_case_index.py is useful for future embeddings,
    but the demo retrieval is deterministic and can run without an embedding
    server.
    """

    settings = get_settings()
    terms = terms or []
    confirmed_features = confirmed_features or []
    denied_features = denied_features or []

    records = get_cases(candidate_ids)
    ranked: list[CaseCandidate] = []
    for case in records:
        score, matched_features, matched_terms = _score_case(
            case, query, terms, confirmed_features, denied_features
        )
        ranked.append(
            CaseCandidate(
                **case.model_dump(),
                score=score,
                matched_features=matched_features,
                matched_terms=matched_terms,
            )
        )

    ranked.sort(key=lambda c: (c.score, -int(c.case_id.split("_")[-1])), reverse=True)
    limit = top_k if top_k is not None else settings.case_initial_top_k
    if limit and limit > 0:
        return ranked[:limit]
    return ranked


def apply_feature_filters(
    candidate_ids: list[str] | None,
    confirmed_features: Iterable[str],
    denied_features: Iterable[str],
) -> list[str]:
    """Narrow candidate ids with confirmed/denied features.

    Filters are conservative: a filter is only applied if it leaves at least one
    candidate. This keeps the demo robust when a user gives a vague or noisy
    answer.
    """

    ids = list(candidate_ids or all_case_ids())
    cmap = case_map()

    for fid in confirmed_features or []:
        narrowed = [cid for cid in ids if fid in cmap[cid].feature_tags]
        if narrowed:
            ids = narrowed

    for fid in denied_features or []:
        narrowed = [cid for cid in ids if fid not in cmap[cid].feature_tags]
        if narrowed:
            ids = narrowed

    return ids


def confidence_from_candidates(candidates: list[CaseCandidate]) -> float:
    if not candidates:
        return 0.0
    if len(candidates) == 1:
        return max(0.65, candidates[0].score)
    gap = max(0.0, candidates[0].score - candidates[1].score)
    base = candidates[0].score
    return round(min(1.0, max(base, 0.5 + gap)), 4)


def top_candidate_summary(candidates: list[CaseCandidate], limit: int | None = None) -> list[dict]:
    settings = get_settings()
    limit = limit or settings.case_display_top_k
    return [
        {
            "case_id": c.case_id,
            "title": c.title,
            "score": c.score,
            "matched_features": c.matched_features,
            "matched_terms": c.matched_terms,
        }
        for c in candidates[:limit]
    ]


def entropy_split_score(pos: int, neg: int, total: int) -> float:
    if total <= 0 or pos <= 0 or neg <= 0:
        return 0.0
    p = pos / total
    # normalized binary entropy, good when split is balanced
    entropy = -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
    return round(entropy, 4)
