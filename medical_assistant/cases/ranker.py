"""★ N-自适应三路评分：sentence_sim + keyword_sim + probe_score。

权重随候选集大小 N 平滑过渡：
  N > 50:  句子级主导（语义粗筛）
  N ≤ 5:   probe + 关键词主导（精细区分）
"""
from __future__ import annotations
from typing import Iterable

from medical_assistant.schemas import CaseRecord, ScoredCase
from medical_assistant.text.embed import cosine, avg_best_match
from medical_assistant.cases.store import (
    get_cases, load_sentence_vecs, load_keyword_vecs,
)


# ── N-自适应权重 ──

def _weights(n: int) -> tuple[float, float, float]:
    """返回 (α_sentence, β_keyword, γ_probe)。"""
    if n > 50:
        return 0.55, 0.20, 0.25
    elif n > 20:
        return 0.40, 0.30, 0.30
    elif n > 5:
        return 0.25, 0.35, 0.40
    else:
        return 0.10, 0.35, 0.55


# ── Probe 奖惩 ──

def _membership(case_id: str, probe_id: str,
                splits: dict[str, dict[str, list[str]]]) -> str:
    s = splits.get(probe_id, {})
    if case_id in set(s.get("positive", [])):
        return "positive"
    if case_id in set(s.get("negative", [])):
        return "negative"
    if case_id in set(s.get("unknown", [])):
        return "unknown"
    return "missing"


def _probe_score(
    case_id: str,
    confirmed: Iterable[str], denied: Iterable[str],
    splits: dict[str, dict[str, list[str]]],
) -> tuple[float, list[str]]:
    score, hits = 0.0, []
    for pid in (confirmed or []):
        m = _membership(case_id, pid, splits)
        if m == "positive":
            score += 0.30; hits.append(pid)
        elif m == "negative":
            score -= 0.40
        elif m == "unknown":
            score += 0.03
    for pid in (denied or []):
        m = _membership(case_id, pid, splits)
        if m == "negative":
            score += 0.22
        elif m == "positive":
            score -= 0.35
        elif m == "unknown":
            score += 0.02
    return score, hits


# ── 句子级相似度 ──

def _sentence_sim(query_vec: list[float], case_id: str,
                  sent_vecs: dict[str, list[float]]) -> float:
    cvec = sent_vecs.get(case_id)
    if not query_vec or not cvec:
        return 0.0
    return max(0.0, cosine(query_vec, cvec))


# ── 关键词级相似度 ──

def _keyword_sim(query_kw_vecs: list[list[float]], case_id: str,
                 kw_vecs: dict[str, dict[str, list[list[float]]]]) -> float:
    """关键词语义匹配：对每个用户关键词，在病例关键词中找最佳匹配。

    正向概念匹配奖励，负向概念匹配惩罚。
    """
    if not query_kw_vecs:
        return 0.0
    case_kw = kw_vecs.get(case_id, {})
    pos_vecs = case_kw.get("positive", [])
    neg_vecs = case_kw.get("negative", [])

    # 正向匹配
    pos_score = avg_best_match(query_kw_vecs, pos_vecs) if pos_vecs else 0.0
    # 负向惩罚：如果用户提到的概念在病例中是"不会出现的"
    neg_score = avg_best_match(query_kw_vecs, neg_vecs) if neg_vecs else 0.0

    return max(0.0, pos_score - 0.4 * neg_score)


# ── 主函数 ──

def rank_cases(
    query_sentence_vec: list[float],
    query_keyword_vecs: list[list[float]],
    candidate_ids: list[str] | None = None,
    confirmed: Iterable[str] = (),
    denied: Iterable[str] = (),
    splits: dict[str, dict[str, list[str]]] | None = None,
) -> list[ScoredCase]:
    """★ N-自适应三路评分排序。

    score = α(N) × sentence_sim + β(N) × keyword_sim + γ(N) × probe_score
    """
    splits = splits or {}
    records = get_cases(candidate_ids)
    N = len(records)
    alpha, beta, gamma = _weights(N)

    sent_vecs = load_sentence_vecs()
    kw_vecs = load_keyword_vecs()

    ranked: list[ScoredCase] = []
    for case in records:
        ss = _sentence_sim(query_sentence_vec, case.case_id, sent_vecs)
        ks = _keyword_sim(query_keyword_vecs, case.case_id, kw_vecs)
        ps, hits = _probe_score(case.case_id, confirmed, denied, splits)

        total = alpha * ss + beta * ks + gamma * max(0.0, ps)
        total = round(max(0.0, min(1.0, total)), 4)

        ranked.append(ScoredCase(
            **case.model_dump(),
            score=total,
            sentence_sim=round(ss, 4),
            keyword_sim=round(ks, 4),
            probe_score=round(ps, 4),
            hit_probes=hits,
        ))

    ranked.sort(key=lambda c: c.score, reverse=True)
    return ranked


def filter_by_probes(
    ids: list[str] | None,
    confirmed: Iterable[str], denied: Iterable[str],
    splits: dict[str, dict[str, list[str]]],
) -> list[str]:
    from medical_assistant.cases.store import all_ids
    result = list(ids or all_ids())
    for pid in (confirmed or []):
        s = splits.get(pid)
        if not s:
            continue
        allowed = set(s.get("positive", []) + s.get("unknown", []))
        narrowed = [c for c in result if c in allowed]
        if narrowed:
            result = narrowed
    for pid in (denied or []):
        s = splits.get(pid)
        if not s:
            continue
        allowed = set(s.get("negative", []) + s.get("unknown", []))
        narrowed = [c for c in result if c in allowed]
        if narrowed:
            result = narrowed
    return result


def confidence(candidates: list[ScoredCase]) -> float:
    if not candidates:
        return 0.0
    if len(candidates) == 1:
        return max(0.65, candidates[0].score)
    gap = max(0.0, candidates[0].score - candidates[1].score)
    return round(min(1.0, max(candidates[0].score, 0.5 + gap)), 4)
