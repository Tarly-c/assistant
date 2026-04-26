"""向量聚类 Window Probe 挖掘。

对病例观察窗口做 TF-IDF 向量聚类，找最能切分当前候选集的 probe。
适合树的中下层或在线动态挖掘。
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

from medical_assistant.schemas import CaseCandidate, CaseRecord
from medical_assistant.text.split import norm_text, split_observation_units
from medical_assistant.text.vectorize import build_idf, cosine, vectorize
from medical_assistant.cases.store import case_extra_texts
from medical_assistant.probes.types import ProbeCandidate, TextUnit
from medical_assistant.probes.scoring import (
    best_threshold_split, make_probe_id, make_probe_label, make_question_seed,
)


def collect_text_units(cases: Sequence[CaseRecord | CaseCandidate]) -> list[TextUnit]:
    units: list[TextUnit] = []
    for case in cases:
        for text in split_observation_units(
            case.title, case.description, extra_texts=case_extra_texts(case),
        ):
            units.append(TextUnit(case_id=case.case_id, text=text))
    return units


def mine_window_probes(
    cases: Sequence[CaseRecord | CaseCandidate], *,
    asked_probe_ids: Iterable[str] | None = None,
    max_probes: int = 5,
    cluster_threshold: float = 0.45,
    probe_prefix: str = "window",
    min_child_size: int = 1,
) -> list[ProbeCandidate]:
    case_ids = [c.case_id for c in cases]
    if len(case_ids) <= 1:
        return []

    units = collect_text_units(cases)
    if not units:
        return []

    idf = build_idf([u.text for u in units])
    unit_vecs = [(u, vectorize(u.text, idf)) for u in units]
    asked = set(asked_probe_ids or [])

    by_case: dict[str, list[tuple[TextUnit, dict[str, float]]]] = {}
    for u, v in unit_vecs:
        by_case.setdefault(u.case_id, []).append((u, v))

    # 简单聚类
    clusters: list[dict[str, Any]] = []
    for u, v in unit_vecs:
        best_idx, best_sim = -1, 0.0
        for idx, cl in enumerate(clusters):
            s = cosine(v, cl["centroid"])
            if s > best_sim:
                best_sim, best_idx = s, idx
        if best_idx >= 0 and best_sim >= cluster_threshold:
            cl = clusters[best_idx]
            cl["members"].append((u, v))
            c = dict(cl["centroid"])
            for k, val in v.items():
                c[k] = c.get(k, 0.0) + val
            n = math.sqrt(sum(x * x for x in c.values())) or 1.0
            cl["centroid"] = {k: x / n for k, x in c.items()}
        else:
            clusters.append({"members": [(u, v)], "centroid": dict(v)})

    candidates: list[ProbeCandidate] = []
    for ci, cl in enumerate(clusters):
        members = cl["members"]
        centroid = cl["centroid"]
        if not members:
            continue
        proto_u, proto_v = max(members, key=lambda x: cosine(x[1], centroid))
        proto_text = proto_u.text
        if len(norm_text(proto_text)) < 4:
            continue
        pid = make_probe_id(probe_prefix, proto_text)
        if pid in asked:
            continue

        sims: dict[str, float] = {}
        ev_by_case: dict[str, str] = {}
        for cid in case_ids:
            bs, bt = 0.0, ""
            for u, v in by_case.get(cid, []):
                s = cosine(proto_v, v)
                if s > bs:
                    bs, bt = s, u.text
            sims[cid] = bs
            ev_by_case[cid] = bt

        positive, negative, unknown, raw_score, th = best_threshold_split(
            sims, min_child_size=min_child_size,
        )
        if not positive or not negative:
            continue

        evidence = [ev_by_case[cid] for cid in positive[:5]
                    if ev_by_case.get(cid) and ev_by_case[cid] not in []]
        evidence = list(dict.fromkeys(evidence))[:5]

        candidates.append(ProbeCandidate(
            probe_id=pid, label=make_probe_label(proto_text),
            prototype_text=proto_text,
            question_seed=make_question_seed(proto_text, evidence),
            positive_case_ids=positive, negative_case_ids=negative,
            unknown_case_ids=unknown, evidence_texts=evidence,
            split_score=round(raw_score, 4),
            debug={"kind": "window_cluster", "threshold": round(th, 4),
                   "positive": len(positive), "negative": len(negative),
                   "unknown": len(unknown), "cluster_index": ci},
        ))

    candidates.sort(key=lambda p: p.split_score, reverse=True)
    return candidates[:max_probes]
