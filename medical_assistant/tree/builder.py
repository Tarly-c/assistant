"""离线建树。用预计算 (K+M) 维特征向量 + 语义切分 + 深度自适应。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from medical_assistant.config import get_settings
from medical_assistant.schemas import CaseRecord
from medical_assistant.cases.store import (
    load_feature_vecs, load_meta, cluster_label, cluster_evidence,
)
from medical_assistant.probes.scoring import semantic_split, split_quality, rephrase


def _unique(ids: Iterable[str]) -> list[str]:
    out, seen = [], set()
    for c in ids:
        if c and c not in seen:
            out.append(c); seen.add(c)
    return out


def _find_best_split(
    case_ids: list[str],
    features: dict[str, list[float]],
    K: int, total_dims: int,
    asked_dims: set[int],
    min_child: int,
    depth: int,
) -> dict[str, Any] | None:
    """在 (K+M) 维特征空间中找最优切分维度。"""
    cfg = get_settings()
    N = len(case_ids)
    if N <= 1 or total_dims == 0:
        return None

    best, best_score = None, -1.0

    for dim in range(total_dims):
        if dim in asked_dims:
            continue
        vals = {cid: features[cid][dim] for cid in case_ids if cid in features}
        if len(vals) < N * 0.8:
            continue

        # ★ 使用 semantic_split
        pos, neg, unk, sq, th = semantic_split(
            vals,
            anchor=cfg.split_anchor,
            search_range=cfg.split_search_range,
            margin=cfg.split_margin,
            min_child=min_child,
        )
        if not pos or not neg or sq <= 0:
            continue

        # 深度自适应权重
        if dim < K:
            weight = max(0.3, 1.0 - depth * 0.12)
        else:
            weight = min(1.5, 0.8 + depth * 0.12)
        adjusted = sq * weight

        if adjusted > best_score:
            best_score = adjusted
            best = {
                "dim": dim, "threshold": round(th, 4),
                "score": round(adjusted, 4), "raw_score": round(sq, 4),
                "weight": round(weight, 3),
                "label": cluster_label(dim),
                "evidence": cluster_evidence(dim),
                "positive_ids": pos, "negative_ids": neg, "unknown_ids": unk,
            }
    return best


def build_tree(cases: Sequence[CaseRecord], *, debug: bool = False) -> dict[str, Any]:
    cfg = get_settings()
    features = load_feature_vecs()
    meta = load_meta()
    K = meta.get("semantic_clusters", 0)
    total_dims = meta.get("total_features", 0)
    case_set = {c.case_id for c in cases}
    nodes: dict[str, dict] = {}

    if not features or total_dims == 0:
        raise RuntimeError("Feature vectors not found. Run scripts/build_vectors.py first.")

    def build(ids: list[str], depth: int, nid: str, asked: set[int]) -> None:
        ids = _unique(c for c in ids if c in case_set and c in features)
        node: dict[str, Any] = {
            "id": nid, "depth": depth, "case_ids": ids,
            "is_leaf": False, "reason": "", "probes": [],
            "primary_dim": -1, "yes_child": "", "no_child": "",
        }
        nodes[nid] = node

        if not ids:
            node.update(is_leaf=True, reason="empty"); return
        if depth >= cfg.tree_max_depth:
            node.update(is_leaf=True, reason="max_depth"); return
        if len(ids) <= cfg.tree_min_leaf:
            node.update(is_leaf=True, reason="small"); return

        mc = 2 if len(ids) >= 8 else 1
        best = _find_best_split(ids, features, K, total_dims, asked, mc, depth)

        if debug and depth <= 1 and best:
            kind = "semantic" if best["dim"] < K else "concept"
            print(f"  [d={depth}] {nid}: dim={best['dim']}({kind}) "
                  f"score={best['score']} raw={best['raw_score']} w={best['weight']} "
                  f"pos={len(best['positive_ids'])} neg={len(best['negative_ids'])} "
                  f"label={best['label']}")

        if not best or best["score"] < cfg.tree_min_gain:
            node.update(is_leaf=True, reason="low_gain" if best else "no_split")
            if best:
                node["rejected"] = {"score": best["score"], "label": best["label"]}
            return

        # LLM 改写追问
        question = rephrase(best["label"], best["evidence"])

        probe = {
            "probe_id": f"dim_{best['dim']}",
            "feature_dim": best["dim"],
            "label": best["label"],
            "question": question,
            "threshold": best["threshold"],
            "score": best["score"],
            "evidence": best["evidence"],
            "positive_ids": best["positive_ids"],
            "negative_ids": best["negative_ids"],
            "unknown_ids": best["unknown_ids"],
            "yes_child": "", "no_child": "",
        }

        soft = cfg.tree_soft_branch
        yes_ids = _unique(best["positive_ids"] + (best["unknown_ids"] if soft else []))
        no_ids = _unique(best["negative_ids"] + (best["unknown_ids"] if soft else []))
        if set(yes_ids) == set(ids):
            yes_ids = best["positive_ids"]
        if set(no_ids) == set(ids):
            no_ids = best["negative_ids"]
        if not yes_ids or not no_ids:
            node.update(is_leaf=True, reason="degenerate"); return

        yc, nc = f"{nid}_y", f"{nid}_n"
        node.update(primary_dim=best["dim"], yes_child=yc, no_child=nc)
        probe["yes_child"], probe["no_child"] = yc, nc
        node["probes"] = [probe]

        # 额外 probe options
        new_asked = asked | {best["dim"]}
        for _ in range(cfg.tree_probes_per_node - 1):
            alt = _find_best_split(ids, features, K, total_dims, new_asked, mc, depth)
            if not alt or alt["score"] < cfg.tree_min_gain * 0.5:
                break
            alt_q = rephrase(alt["label"], alt["evidence"])
            node["probes"].append({
                "probe_id": f"dim_{alt['dim']}",
                "feature_dim": alt["dim"],
                "label": alt["label"], "question": alt_q,
                "threshold": alt["threshold"], "score": alt["score"],
                "evidence": alt["evidence"],
                "positive_ids": alt["positive_ids"],
                "negative_ids": alt["negative_ids"],
                "unknown_ids": alt["unknown_ids"],
                "yes_child": "", "no_child": "",
            })
            new_asked = new_asked | {alt["dim"]}

        build(yes_ids, depth + 1, yc, asked | {best["dim"]})
        build(no_ids, depth + 1, nc, asked | {best["dim"]})

    build([c.case_id for c in cases if c.case_id in features], 0, "n", set())
    return {
        "version": 5, "root": "n", "count": len(cases),
        "K": K, "M": total_dims - K, "total_dims": total_dims,
        "split_config": {
            "anchor": cfg.split_anchor,
            "search_range": cfg.split_search_range,
            "margin": cfg.split_margin,
        },
        "nodes": nodes,
    }


def save_tree(tree: dict, path: Path | None = None) -> Path:
    path = path or get_settings().tree_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tree, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def tree_stats(tree: dict) -> dict:
    ns = tree.get("nodes", {})
    leaves = sum(1 for n in ns.values() if isinstance(n, dict) and n.get("is_leaf"))
    mx = max((int(n.get("depth", 0)) for n in ns.values() if isinstance(n, dict)), default=0)
    return {"nodes": len(ns), "internal": len(ns) - leaves, "leaves": leaves, "max_depth": mx}
