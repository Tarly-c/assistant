"""子句级 Anchor Probe 挖掘。

从病例描述中切子句，用子串包含判断哪些病例匹配。
无 unknown 桶，切分干净。适合树的上层。
"""
from __future__ import annotations

import re
from typing import Iterable, Sequence

from medical_assistant.schemas import CaseCandidate, CaseRecord
from medical_assistant.text.split import norm_text, split_clauses
from medical_assistant.cases.store import case_search_text
from medical_assistant.probes.types import ProbeCandidate
from medical_assistant.probes.scoring import (
    make_probe_id, make_probe_label, make_question_seed, split_quality,
)


def _extract_clause_keys(clause: str) -> list[str]:
    """从子句中提取 2-4 字核心片段，用于跨病例子串匹配。"""
    compact = norm_text(re.sub(r"[^\u4e00-\u9fffa-z0-9]", "", clause))
    if len(compact) < 2:
        return []
    keys: list[str] = []
    if 2 <= len(compact) <= 12:
        keys.append(compact)
    for n in (4, 3, 2):
        for i in range(len(compact) - n + 1):
            g = compact[i: i + n]
            if g not in keys:
                keys.append(g)
    return keys


def _clause_matches_case(keys: list[str], case_text_norm: str) -> bool:
    if not keys:
        return False
    return keys[0] in case_text_norm


def mine_anchor_probes(
    cases: Sequence[CaseRecord | CaseCandidate], *,
    asked_probe_ids: Iterable[str] | None = None,
    max_probes: int = 10,
    probe_prefix: str = "anchor",
    min_child_size: int = 2,
    min_child_ratio: float = 0.05,
) -> list[ProbeCandidate]:
    if len(cases) <= 1:
        return []

    total = len(cases)
    asked = set(asked_probe_ids or [])
    min_child = max(1, min_child_size, int(total * min_child_ratio))

    case_texts_norm = {c.case_id: norm_text(case_search_text(c)) for c in cases}

    clause_keys_cache: dict[str, list[str]] = {}
    clause_source: dict[str, str] = {}
    clause_to_case_ids: dict[str, set[str]] = {}

    for case in cases:
        for clause in split_clauses(case.title) + split_clauses(case.description):
            cn = norm_text(clause)
            if len(cn) < 4 or len(cn) > 60:
                continue
            if cn not in clause_keys_cache:
                clause_keys_cache[cn] = _extract_clause_keys(clause)
                clause_source[cn] = clause
            clause_to_case_ids.setdefault(cn, set()).add(case.case_id)

    candidates: list[ProbeCandidate] = []
    for cn, keys in clause_keys_cache.items():
        if not keys:
            continue
        pid = make_probe_id(probe_prefix, cn)
        if pid in asked:
            continue

        positive = [c.case_id for c in cases
                    if _clause_matches_case(keys, case_texts_norm[c.case_id])]
        negative = [c.case_id for c in cases if c.case_id not in set(positive)]

        if len(positive) < min_child or len(negative) < min_child:
            continue
        score = split_quality(len(positive), len(negative), 0, total)
        if score <= 0.0:
            continue

        original = clause_source.get(cn, cn)
        evidence = [original]
        for cid in positive[:5]:
            obj = next((c for c in cases if c.case_id == cid), None)
            if obj:
                for u in split_clauses(obj.description)[:3]:
                    if norm_text(keys[0]) in norm_text(u) and u not in evidence:
                        evidence.append(u)
                        break
            if len(evidence) >= 4:
                break

        candidates.append(ProbeCandidate(
            probe_id=pid,
            label=make_probe_label(original),
            prototype_text=original,
            question_seed=make_question_seed(original, evidence),
            positive_case_ids=positive,
            negative_case_ids=negative,
            unknown_case_ids=[],
            evidence_texts=evidence[:5],
            split_score=round(score, 4),
            debug={"kind": "anchor_clause", "primary_key": keys[0],
                   "positive": len(positive), "negative": len(negative), "unknown": 0},
        ))

    candidates.sort(key=lambda p: p.split_score, reverse=True)
    seen: set[frozenset[str]] = set()
    deduped: list[ProbeCandidate] = []
    for p in candidates:
        sig = frozenset(p.positive_case_ids)
        if sig not in seen:
            seen.add(sig)
            deduped.append(p)
        if len(deduped) >= max_probes:
            break
    return deduped
