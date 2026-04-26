from __future__ import annotations

import json
from functools import lru_cache
from typing import Iterable

from medical_assistant.config import get_settings
from medical_assistant.schemas import CaseCandidate, CaseRecord
from medical_assistant.services.cases.features import extract_search_terms


def make_case_id(index: int) -> str:
    return f"case_{index:03d}"


def _list_field(item: dict, *names: str) -> list[str]:
    out: list[str] = []
    for name in names:
        value = item.get(name)
        if isinstance(value, str):
            if value.strip():
                out.append(value.strip())
        elif isinstance(value, list):
            out.extend(str(x).strip() for x in value if str(x).strip())
    seen: set[str] = set()
    deduped: list[str] = []
    for value in out:
        if value not in seen:
            deduped.append(value)
            seen.add(value)
    return deduped


def case_document_text(case: CaseRecord) -> str:
    parts = [
        f"标题: {case.title}",
        f"描述: {case.description}",
        f"处理: {case.treat}",
    ]
    if case.title_en:
        parts.append(f"Title EN: {case.title_en}")
    if case.description_en:
        parts.append(f"Description EN: {case.description_en}")
    extra = [*case.aliases, *case.keywords, *case.key_terms_en, *case.search_terms, *case.feature_tags]
    if extra:
        parts.append("Search terms: " + " / ".join(extra))
    return "\n".join(part for part in parts if part.strip())


@lru_cache(maxsize=1)
def load_cases() -> list[CaseRecord]:
    settings = get_settings()
    path = settings.case_data_path
    if not path.exists():
        raise FileNotFoundError(
            f"Case data not found: {path}. Put the JSON file at this path or set MEDICAL_ASSISTANT_CASE_DATA_FILE."
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Case data must be a JSON list")

    cases: list[CaseRecord] = []
    for index, item in enumerate(raw, 1):
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        description = str(item.get("description", "")).strip()
        treat = str(item.get("treat") or item.get("treatment") or "").strip()
        cases.append(
            CaseRecord(
                case_id=str(item.get("case_id") or make_case_id(index)),
                title=title,
                description=description,
                treat=treat,
                title_en=str(item.get("title_en") or "").strip(),
                description_en=str(item.get("description_en") or "").strip(),
                aliases=_list_field(item, "aliases", "alias", "synonyms"),
                keywords=_list_field(item, "keywords", "keyword"),
                key_terms_en=_list_field(item, "key_terms_en", "terms_en", "en_terms"),
                search_terms=_list_field(item, "search_terms", "search_aliases"),
                feature_tags=_list_field(item, "feature_tags"),
            )
        )
    return cases


@lru_cache(maxsize=1)
def case_map() -> dict[str, CaseRecord]:
    return {case.case_id: case for case in load_cases()}


def clear_case_cache() -> None:
    load_cases.cache_clear()
    case_map.cache_clear()


def get_case(case_id: str) -> CaseRecord | None:
    return case_map().get(case_id)


def get_cases(case_ids: Iterable[str] | None = None) -> list[CaseRecord]:
    if case_ids is None:
        return load_cases()
    cmap = case_map()
    return [cmap[cid] for cid in case_ids if cid in cmap]


def all_case_ids() -> list[str]:
    return [case.case_id for case in load_cases()]


def _norm(text: str) -> str:
    return (text or "").lower().strip()


def _term_hits(text: str, terms: Iterable[str]) -> list[str]:
    haystack = _norm(text)
    hits: list[str] = []
    for term in terms:
        t = _norm(term)
        if t and t in haystack and term not in hits:
            hits.append(term)
    return hits


def _query_is_too_short(query: str, terms: list[str]) -> bool:
    q = _norm(query)
    return (not q or len(q) <= 1) and not terms


def _split_membership(
    case_id: str,
    probe_id: str,
    probe_splits: dict[str, dict[str, list[str]]] | None,
) -> str:
    split = (probe_splits or {}).get(probe_id) or {}
    if case_id in set(split.get("positive", [])):
        return "positive"
    if case_id in set(split.get("negative", [])):
        return "negative"
    if case_id in set(split.get("unknown", [])):
        return "unknown"
    return "missing"


def _score_case(
    case: CaseRecord,
    query: str,
    terms: list[str],
    confirmed_features: Iterable[str],
    denied_features: Iterable[str],
    probe_splits: dict[str, dict[str, list[str]]] | None = None,
) -> tuple[float, list[str], list[str]]:
    text = case_document_text(case)
    title_text = "\n".join([case.title, case.title_en, *case.aliases, *case.key_terms_en])
    confirmed = list(confirmed_features or [])
    denied = list(denied_features or [])

    search_terms = [query] if query and not _query_is_too_short(query, terms) else []
    for term in terms or []:
        if term not in search_terms:
            search_terms.append(term)
    for term in extract_search_terms(query):
        if term not in search_terms:
            search_terms.append(term)

    matched_terms = _term_hits(text, search_terms)
    title_hits = _term_hits(title_text, search_terms)
    score = 0.0
    if search_terms:
        score += 0.35 * (len(matched_terms) / max(1, len(search_terms)))
        score += 0.15 * (len(title_hits) / max(1, len(search_terms)))

    matched_features: list[str] = []
    for probe_id in confirmed:
        membership = _split_membership(case.case_id, probe_id, probe_splits)
        if membership == "positive":
            score += 0.24
            matched_features.append(probe_id)
        elif membership == "negative":
            score -= 0.35
        elif membership == "unknown":
            score += 0.03

    for probe_id in denied:
        membership = _split_membership(case.case_id, probe_id, probe_splits)
        if membership == "negative":
            score += 0.18
        elif membership == "positive":
            score -= 0.32
        elif membership == "unknown":
            score += 0.02

    score = max(0.0, min(1.0, score))
    return round(score, 4), matched_features, matched_terms


def search_cases(
    query: str,
    terms: list[str] | None = None,
    candidate_ids: list[str] | None = None,
    confirmed_features: Iterable[str] | None = None,
    denied_features: Iterable[str] | None = None,
    probe_splits: dict[str, dict[str, list[str]]] | None = None,
    top_k: int | None = None,
) -> list[CaseCandidate]:
    settings = get_settings()
    terms = terms or []
    confirmed_features = confirmed_features or []
    denied_features = denied_features or []
    records = get_cases(candidate_ids)
    ranked: list[CaseCandidate] = []
    for case in records:
        score, matched_features, matched_terms = _score_case(
            case,
            query,
            terms,
            confirmed_features,
            denied_features,
            probe_splits=probe_splits,
        )
        ranked.append(
            CaseCandidate(
                **case.model_dump(),
                score=score,
                matched_features=matched_features,
                matched_terms=matched_terms,
            )
        )
    ranked.sort(key=lambda c: (c.score, -int(c.case_id.split("_")[-1]) if "_" in c.case_id else 0), reverse=True)
    limit = top_k if top_k is not None else settings.case_initial_top_k
    if limit and limit > 0:
        return ranked[:limit]
    return ranked


def _allowed_from_split(
    split: dict[str, list[str]],
    *,
    signal: str,
    keep_unknown: bool = True,
) -> list[str]:
    if signal == "yes":
        ids = list(split.get("positive", []))
    else:
        ids = list(split.get("negative", []))
    if keep_unknown:
        ids.extend(split.get("unknown", []))
    seen: set[str] = set()
    out: list[str] = []
    for cid in ids:
        if cid and cid not in seen:
            out.append(cid)
            seen.add(cid)
    return out


def apply_feature_filters(
    candidate_ids: list[str] | None,
    confirmed_features: Iterable[str],
    denied_features: Iterable[str],
    probe_splits: dict[str, dict[str, list[str]]] | None = None,
) -> list[str]:
    ids = list(candidate_ids or all_case_ids())
    splits = probe_splits or {}
    for probe_id in confirmed_features or []:
        split = splits.get(probe_id)
        if not split:
            continue
        allowed = set(_allowed_from_split(split, signal="yes", keep_unknown=True))
        narrowed = [cid for cid in ids if cid in allowed]
        if narrowed:
            ids = narrowed
    for probe_id in denied_features or []:
        split = splits.get(probe_id)
        if not split:
            continue
        allowed = set(_allowed_from_split(split, signal="no", keep_unknown=True))
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
    base = candidates[0].score
    return round(min(1.0, max(base, 0.5 + gap)), 4)


def top_candidate_summary(candidates: list[CaseCandidate], limit: int | None = None) -> list[dict]:
    settings = get_settings()
    limit = limit or settings.case_display_top_k
    return [
        {
            "case_id": case.case_id,
            "title": case.title,
            "score": case.score,
            "matched_features": case.matched_features,
            "matched_terms": case.matched_terms,
        }
        for case in candidates[:limit]
    ]
