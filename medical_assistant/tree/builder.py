"""离线构建问诊决策树。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from medical_assistant.config import get_settings
from medical_assistant.schemas import CaseRecord
from medical_assistant.probes.types import ProbeCandidate
from medical_assistant.probes.mine import mine_probes


def _unique(ids: Iterable[str]) -> list[str]:
    out, seen = [], set()
    for c in ids:
        if c and c not in seen:
            out.append(c); seen.add(c)
    return out


def _probe_dict(p: ProbeCandidate) -> dict[str, Any]:
    return {
        "probe_id": p.probe_id, "label": p.label, "prototype": p.prototype,
        "question": p.question, "positive_ids": p.positive_ids,
        "negative_ids": p.negative_ids, "unknown_ids": p.unknown_ids,
        "evidence": p.evidence, "score": p.score, "debug": p.debug,
        "yes_child": "", "no_child": "",
    }


def _branch(probe: dict, side: str, *, soft: bool) -> list[str]:
    ids = list(probe.get("positive_ids" if side == "yes" else "negative_ids", []))
    if soft:
        ids.extend(probe.get("unknown_ids", []))
    return _unique(ids)


def build_tree(cases: Sequence[CaseRecord], *, debug: bool = False) -> dict[str, Any]:
    """递归构建问诊决策树。每个节点的 probe 从当前病例子集中自动挖掘。"""
    cfg = get_settings()
    case_map = {c.case_id: c for c in cases}
    nodes: dict[str, dict] = {}

    def build(ids: list[str], depth: int, nid: str, asked: list[str]) -> None:
        ids = _unique(c for c in ids if c in case_map)
        node = {"id": nid, "depth": depth, "case_ids": ids,
                "is_leaf": False, "reason": "", "probes": [],
                "primary": "", "yes_child": "", "no_child": "", "leaf_ids": []}
        nodes[nid] = node

        # 终止条件
        if not ids:
            node.update(is_leaf=True, reason="empty", leaf_ids=ids); return
        if depth >= cfg.tree_max_depth:
            node.update(is_leaf=True, reason="max_depth", leaf_ids=ids); return
        if len(ids) <= cfg.tree_min_leaf:
            node.update(is_leaf=True, reason="small", leaf_ids=ids); return

        # 挖掘 probe
        subset = [case_map[c] for c in ids]
        probes = mine_probes(
            subset, asked=asked, max_probes=max(1, cfg.tree_probes_per_node),
            prefix=f"{nid}_p", min_child=2 if len(ids) >= 8 else 1,
        )

        if debug and depth == 0:
            print(f"  [build] root: {len(probes)} candidates")
            for p in probes[:5]:
                print(f"    score={p.score:.4f} pos={len(p.positive_ids)} "
                      f"neg={len(p.negative_ids)} label={p.label}")

        if not probes:
            node.update(is_leaf=True, reason="no_probe", leaf_ids=ids); return

        pd = [_probe_dict(p) for p in probes]
        node["probes"] = pd
        best = pd[0]

        if best["score"] < cfg.tree_min_gain:
            node.update(is_leaf=True, reason="low_gain", leaf_ids=ids,
                        rejected={"score": best["score"], "label": best["label"]})
            return

        # 切分
        yes_ids = _branch(best, "yes", soft=cfg.tree_soft_branch)
        no_ids = _branch(best, "no", soft=cfg.tree_soft_branch)
        if set(yes_ids) == set(ids):
            yes_ids = _branch(best, "yes", soft=False)
        if set(no_ids) == set(ids):
            no_ids = _branch(best, "no", soft=False)
        if not yes_ids or not no_ids:
            node.update(is_leaf=True, reason="degenerate", leaf_ids=ids); return

        yc, nc = f"{nid}_y", f"{nid}_n"
        node.update(primary=best["probe_id"], yes_child=yc, no_child=nc)
        best["yes_child"], best["no_child"] = yc, nc

        build(yes_ids, depth + 1, yc, asked + [best["probe_id"]])
        build(no_ids, depth + 1, nc, asked + [best["probe_id"]])

    build([c.case_id for c in cases], 0, "n", [])
    return {"version": 3, "root": "n", "count": len(cases),
            "config": {"max_depth": cfg.tree_max_depth, "min_leaf": cfg.tree_min_leaf,
                       "min_gain": cfg.tree_min_gain, "probes_per_node": cfg.tree_probes_per_node,
                       "soft_branch": cfg.tree_soft_branch},
            "nodes": nodes}


def save_tree(tree: dict, path: Path | None = None) -> Path:
    path = path or get_settings().tree_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tree, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def tree_stats(tree: dict) -> dict[str, int]:
    ns = tree.get("nodes", {})
    leaves = sum(1 for n in ns.values() if isinstance(n, dict) and n.get("is_leaf"))
    mx = max((int(n.get("depth", 0)) for n in ns.values() if isinstance(n, dict)), default=0)
    return {"nodes": len(ns), "internal": len(ns) - leaves, "leaves": leaves, "max_depth": mx}
