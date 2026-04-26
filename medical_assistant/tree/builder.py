"""离线构建问诊决策树。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from medical_assistant.config import get_settings
from medical_assistant.schemas import CaseRecord
from medical_assistant.probes.types import ProbeCandidate
from medical_assistant.probes.mine import mine_tree_probes

TREE_VERSION = 3


def _unique(ids: Iterable[str]) -> list[str]:
    out, seen = [], set()
    for cid in ids:
        if cid and cid not in seen:
            out.append(cid)
            seen.add(cid)
    return out


def _probe_to_dict(p: ProbeCandidate) -> dict[str, Any]:
    return {
        "probe_id": p.probe_id, "label": p.label,
        "prototype_text": p.prototype_text, "question_seed": p.question_seed,
        "positive_case_ids": p.positive_case_ids,
        "negative_case_ids": p.negative_case_ids,
        "unknown_case_ids": p.unknown_case_ids,
        "evidence_texts": p.evidence_texts, "split_score": p.split_score,
        "debug": p.debug, "yes_child_id": "", "no_child_id": "",
    }


def _branch_ids(probe: dict, branch: str, *, include_unknown: bool) -> list[str]:
    ids = list(probe.get("positive_case_ids" if branch == "yes" else "negative_case_ids", []))
    if include_unknown:
        ids.extend(probe.get("unknown_case_ids", []))
    return _unique(ids)


def _mark_leaf(node: dict, reason: str, case_ids: list[str]) -> None:
    node["is_leaf"] = True
    node["stop_reason"] = reason
    node["leaf_case_ids"] = case_ids


def build_question_tree(cases: Sequence[CaseRecord], *, debug: bool = False) -> dict[str, Any]:
    settings = get_settings()
    case_map = {c.case_id: c for c in cases}
    nodes: dict[str, dict[str, Any]] = {}

    def build(case_ids: list[str], depth: int, nid: str, asked: list[str]) -> str:
        case_ids = _unique(cid for cid in case_ids if cid in case_map)
        node: dict[str, Any] = {
            "node_id": nid, "depth": depth, "case_ids": case_ids,
            "is_leaf": False, "stop_reason": "", "probe_options": [],
            "primary_probe_id": "", "yes_child_id": "", "no_child_id": "",
            "leaf_case_ids": [],
        }
        nodes[nid] = node

        if not case_ids:
            _mark_leaf(node, "empty", case_ids); return nid
        if depth >= settings.tree_max_depth:
            _mark_leaf(node, "max_depth", case_ids); return nid
        if len(case_ids) <= settings.tree_min_leaf_cases:
            _mark_leaf(node, "small_leaf", case_ids); return nid

        subset = [case_map[cid] for cid in case_ids]
        min_cs = 2 if len(case_ids) >= 8 else 1
        probes = mine_tree_probes(
            subset, asked_probe_ids=asked,
            max_probes=max(1, settings.tree_probe_options_per_node),
            probe_prefix=f"{nid}_p", min_child_size=min_cs,
            min_child_ratio=0.05 if len(case_ids) >= 8 else 0.0,
        )

        if debug and depth == 0:
            print(f"  [build] root candidates: {len(probes)}")
            for p in probes[:5]:
                print(f"    score={p.split_score:.4f} label={p.label} "
                      f"pos={len(p.positive_case_ids)} neg={len(p.negative_case_ids)} "
                      f"kind={p.debug.get('kind','')}")

        if not probes:
            _mark_leaf(node, "no_probe", case_ids); return nid

        probe_dicts = [_probe_to_dict(p) for p in probes]
        node["probe_options"] = probe_dicts
        primary = probe_dicts[0]
        ps = float(primary.get("split_score", 0) or 0)

        if ps < settings.tree_min_probe_gain:
            node["best_rejected"] = {"score": ps, "label": primary.get("label", ""),
                                     "debug": primary.get("debug", {})}
            _mark_leaf(node, "low_gain", case_ids); return nid

        use_unk = settings.tree_use_unknown_as_soft_branch
        yes_ids = _branch_ids(primary, "yes", include_unknown=use_unk)
        no_ids = _branch_ids(primary, "no", include_unknown=use_unk)
        if set(yes_ids) == set(case_ids):
            yes_ids = _branch_ids(primary, "yes", include_unknown=False)
        if set(no_ids) == set(case_ids):
            no_ids = _branch_ids(primary, "no", include_unknown=False)
        if not yes_ids or not no_ids:
            node["best_rejected"] = {"score": ps, "label": primary.get("label", ""),
                                     "debug": primary.get("debug", {})}
            _mark_leaf(node, "degenerate_split", case_ids); return nid

        node["primary_probe_id"] = primary["probe_id"]
        yc, nc = f"{nid}_y", f"{nid}_n"
        node["yes_child_id"], node["no_child_id"] = yc, nc
        primary["yes_child_id"], primary["no_child_id"] = yc, nc

        next_asked = asked + [str(primary["probe_id"])]
        build(yes_ids, depth + 1, yc, next_asked)
        build(no_ids, depth + 1, nc, next_asked)
        return nid

    build([c.case_id for c in cases], 0, "n", [])
    return {
        "version": TREE_VERSION, "root_id": "n",
        "case_count": len(cases),
        "settings": {
            "tree_max_depth": settings.tree_max_depth,
            "tree_min_leaf_cases": settings.tree_min_leaf_cases,
            "tree_min_probe_gain": settings.tree_min_probe_gain,
            "tree_probe_options_per_node": settings.tree_probe_options_per_node,
            "tree_use_unknown_as_soft_branch": settings.tree_use_unknown_as_soft_branch,
        },
        "nodes": nodes,
    }


def save_question_tree(tree: dict[str, Any], path: Path | None = None) -> Path:
    path = path or get_settings().case_question_tree_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tree, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def tree_stats(tree: dict[str, Any]) -> dict[str, int]:
    nodes = tree.get("nodes", {}) if isinstance(tree, dict) else {}
    leaves = sum(1 for n in nodes.values() if isinstance(n, dict) and n.get("is_leaf"))
    internals = len(nodes) - leaves
    mx = max((int(n.get("depth", 0)) for n in nodes.values() if isinstance(n, dict)), default=0)
    return {"nodes": len(nodes), "internal": internals, "leaves": leaves, "max_depth": mx}
