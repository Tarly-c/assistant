"""向量聚类 Window Probe 挖掘。"""
from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

from medical_assistant.schemas import CaseRecord, ScoredCase
from medical_assistant.text.split import norm, split_windows
from medical_assistant.text.vectorize import build_idf, cosine, vectorize
from medical_assistant.cases.store import extra_texts
from medical_assistant.probes.types import ProbeCandidate, TextUnit
from medical_assistant.probes.scoring import best_threshold, probe_id, probe_label, probe_question


def _collect_units(cases: Sequence[CaseRecord | ScoredCase]) -> list[TextUnit]:
    units = []
    for c in cases:
        for t in split_windows(c.title, c.description, extra=extra_texts(c)):
            units.append(TextUnit(case_id=c.case_id, text=t))
    return units


def mine_window_probes(
    cases: Sequence[CaseRecord | ScoredCase], *,
    asked: Iterable[str] = (), max_probes: int = 5,
    prefix: str = "w", min_child: int = 1, cluster_th: float = 0.45,
) -> list[ProbeCandidate]:
    ids = [c.case_id for c in cases]
    if len(ids) <= 1:
        return []
    units = _collect_units(cases)
    if not units:
        return []

    idf = build_idf([u.text for u in units])
    uv = [(u, vectorize(u.text, idf)) for u in units]
    asked_set = set(asked)

    # 按病例索引
    by_case: dict[str, list[tuple[TextUnit, dict]]] = {}
    for u, v in uv:
        by_case.setdefault(u.case_id, []).append((u, v))

    # 简单在线聚类
    clusters: list[dict[str, Any]] = []
    for u, v in uv:
        bi, bs = -1, 0.0
        for i, cl in enumerate(clusters):
            s = cosine(v, cl["c"])
            if s > bs:
                bs, bi = s, i
        if bi >= 0 and bs >= cluster_th:
            cl = clusters[bi]
            cl["m"].append((u, v))
            c = dict(cl["c"])
            for k, val in v.items():
                c[k] = c.get(k, 0.0) + val
            n = math.sqrt(sum(x * x for x in c.values())) or 1.0
            cl["c"] = {k: x / n for k, x in c.items()}
        else:
            clusters.append({"m": [(u, v)], "c": dict(v)})

    results: list[ProbeCandidate] = []
    for ci, cl in enumerate(clusters):
        if not cl["m"]:
            continue
        proto_u, proto_v = max(cl["m"], key=lambda x: cosine(x[1], cl["c"]))
        if len(norm(proto_u.text)) < 4:
            continue
        pid = probe_id(prefix, proto_u.text)
        if pid in asked_set:
            continue

        sims, ev_map = {}, {}
        for cid in ids:
            bs, bt = 0.0, ""
            for u, v in by_case.get(cid, []):
                s = cosine(proto_v, v)
                if s > bs:
                    bs, bt = s, u.text
            sims[cid] = bs
            ev_map[cid] = bt

        pos, neg, unk, sq, th = best_threshold(sims, min_child=min_child)
        if not pos or not neg:
            continue
        ev = list(dict.fromkeys(ev_map[c] for c in pos[:5] if ev_map.get(c)))[:5]
        results.append(ProbeCandidate(
            probe_id=pid, label=probe_label(proto_u.text), prototype=proto_u.text,
            question=probe_question(proto_u.text, ev),
            positive_ids=pos, negative_ids=neg, unknown_ids=unk,
            evidence=ev, score=round(sq, 4),
            debug={"kind": "window", "threshold": round(th, 4),
                   "pos": len(pos), "neg": len(neg), "unk": len(unk), "ci": ci},
        ))

    results.sort(key=lambda p: p.score, reverse=True)
    return results[:max_probes]
