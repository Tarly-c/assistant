"""统一 Probe 挖掘：基于预计算 (K+M) 维特征向量 + 语义切分。

遍历特征维度 → semantic_split 找最优切分 → LLM 改写追问。
"""
from __future__ import annotations
from typing import Iterable, Sequence

from medical_assistant.config import get_settings
from medical_assistant.cases.store import (
    load_feature_vecs, load_meta, cluster_label, cluster_evidence,
)
from medical_assistant.probes.types import ProbeCandidate
from medical_assistant.probes.scoring import semantic_split, rephrase


def _depth_weight(dim: int, K: int, depth_hint: int) -> float:
    """深度自适应：浅层偏好语义维度，深层偏好概念维度。"""
    if dim < K:
        return max(0.3, 1.0 - depth_hint * 0.12)
    else:
        return min(1.5, 0.8 + depth_hint * 0.12)


def mine_probes(
    candidate_ids: Sequence[str], *,
    asked: Iterable[str] = (),
    max_probes: int = 5,
    min_child: int = 1,
    depth_hint: int = 0,
) -> list[ProbeCandidate]:
    """在当前候选集中寻找最佳切分维度。"""
    cfg = get_settings()
    features = load_feature_vecs()
    meta = load_meta()
    K = meta.get("semantic_clusters", 0)
    total_dims = meta.get("total_features", 0)

    if not features or total_dims == 0:
        return []

    ids = [c for c in candidate_ids if c in features]
    N = len(ids)
    if N <= 1:
        return []

    mc = max(1, min_child, int(N * 0.05))
    asked_set = set(asked)

    # 已问过的维度
    asked_dims: set[int] = set()
    for pid in asked_set:
        if pid.startswith("dim_"):
            try:
                asked_dims.add(int(pid.split("_")[1]))
            except (ValueError, IndexError):
                pass

    candidates: list[ProbeCandidate] = []

    for dim in range(total_dims):
        pid = f"dim_{dim}"
        if pid in asked_set or dim in asked_dims:
            continue

        vals = {cid: features[cid][dim] for cid in ids}

        # ★ 使用 semantic_split 替代 best_threshold
        pos, neg, unk, sq, th = semantic_split(
            vals,
            anchor=cfg.split_anchor,
            search_range=cfg.split_search_range,
            margin=cfg.split_margin,
            min_child=mc,
        )
        if not pos or not neg or sq <= 0:
            continue

        adjusted = sq * _depth_weight(dim, K, depth_hint)

        label = cluster_label(dim)
        evidence = cluster_evidence(dim)

        candidates.append(ProbeCandidate(
            probe_id=pid,
            feature_dim=dim,
            label=label,
            question="",
            positive_ids=pos,
            negative_ids=neg,
            unknown_ids=unk,
            evidence=evidence,
            score=round(adjusted, 4),
            debug={
                "kind": "semantic" if dim < K else "concept",
                "raw_score": round(sq, 4),
                "depth_weight": round(_depth_weight(dim, K, depth_hint), 3),
                "threshold": round(th, 4),
                "anchor": cfg.split_anchor,
                "pos": len(pos), "neg": len(neg), "unk": len(unk),
            },
        ))

    candidates.sort(key=lambda p: p.score, reverse=True)
    result = candidates[:max_probes]

    # 只对 top 做 LLM 改写
    for p in result[:2]:
        p.question = rephrase(p.label, p.evidence)

    return result
