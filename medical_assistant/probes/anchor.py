"""子句级 Anchor Probe 挖掘。子串包含匹配，无 unknown。"""
from __future__ import annotations

import re
from typing import Iterable, Sequence

from medical_assistant.schemas import CaseRecord, ScoredCase
from medical_assistant.text.split import norm, split_clauses
from medical_assistant.cases.store import full_text
from medical_assistant.probes.types import ProbeCandidate
from medical_assistant.probes.scoring import probe_id, probe_label, probe_question, split_quality


def _clause_keys(clause: str) -> list[str]:
    """提取子句的 2-12 字核心片段，用于子串匹配。"""
    c = norm(re.sub(r"[^\u4e00-\u9fffa-z0-9]", "", clause))
    if len(c) < 2:
        return []
    keys = []
    if 2 <= len(c) <= 12:
        keys.append(c)
    for n in (4, 3, 2):
        for i in range(len(c) - n + 1):
            g = c[i: i + n]
            if g not in keys:
                keys.append(g)
    return keys


def mine_anchor_probes(
    cases: Sequence[CaseRecord | ScoredCase], *,
    asked: Iterable[str] = (), max_probes: int = 10,
    prefix: str = "a", min_child: int = 2,
) -> list[ProbeCandidate]:
    if len(cases) <= 1:
        return []
    total = len(cases)
    asked_set = set(asked)
    texts_norm = {c.case_id: norm(full_text(c)) for c in cases}

    # 收集所有子句及其核心片段
    cache: dict[str, list[str]] = {}     # clause_norm → keys
    source: dict[str, str] = {}          # clause_norm → 原始文本

    for case in cases:
        for clause in split_clauses(case.title) + split_clauses(case.description):
            cn = norm(clause)
            if 4 <= len(cn) <= 60 and cn not in cache:
                cache[cn] = _clause_keys(clause)
                source[cn] = clause

    results: list[ProbeCandidate] = []
    for cn, keys in cache.items():
        if not keys:
            continue
        pid = probe_id(prefix, cn)
        if pid in asked_set:
            continue

        pos = [c.case_id for c in cases if keys[0] in texts_norm[c.case_id]]
        neg = [c.case_id for c in cases if c.case_id not in set(pos)]
        if len(pos) < min_child or len(neg) < min_child:
            continue

        sq = split_quality(len(pos), len(neg), 0, total)
        if sq <= 0:
            continue

        orig = source.get(cn, cn)
        ev = [orig]
        for cid in pos[:4]:
            obj = next((c for c in cases if c.case_id == cid), None)
            if obj:
                for u in split_clauses(obj.description)[:3]:
                    if norm(keys[0]) in norm(u) and u not in ev:
                        ev.append(u); break
            if len(ev) >= 4:
                break

        results.append(ProbeCandidate(
            probe_id=pid, label=probe_label(orig), prototype=orig,
            question=probe_question(orig, ev),
            positive_ids=pos, negative_ids=neg, evidence=ev[:5],
            score=round(sq, 4),
            debug={"kind": "anchor", "key": keys[0], "pos": len(pos), "neg": len(neg)},
        ))

    results.sort(key=lambda p: p.score, reverse=True)
    # positive set 去重
    seen: set[frozenset[str]] = set()
    deduped = []
    for p in results:
        sig = frozenset(p.positive_ids)
        if sig not in seen:
            seen.add(sig)
            deduped.append(p)
        if len(deduped) >= max_probes:
            break
    return deduped
