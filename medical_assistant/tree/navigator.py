"""在线沿离线决策树选择下一个追问。"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Sequence

from medical_assistant.config import get_settings
from medical_assistant.schemas import Memory, Probe, ScoredCase
from medical_assistant.probes.scoring import split_quality


@lru_cache(maxsize=1)
def load_tree() -> dict[str, Any] | None:
    path = get_settings().tree_path
    if not path.exists():
        return None
    try:
        t = json.loads(path.read_text(encoding="utf-8"))
        return t if isinstance(t, dict) and "nodes" in t else None
    except Exception:
        return None


def clear_cache() -> None:
    load_tree.cache_clear()


def _get_node(tree: dict, nid: str) -> dict | None:
    n = tree.get("nodes", {}).get(nid)
    return n if isinstance(n, dict) else None


def _locate(tree: dict, cids: list[str]) -> str:
    """找到最匹配当前候选集的树节点。"""
    root = str(tree.get("root", "n"))
    if not cids:
        return root
    target = set(cids)
    best_id, best_t = root, (-1.0, -1, 10**9)
    for nid, node in tree.get("nodes", {}).items():
        nc = set(node.get("case_ids", []))
        if not nc:
            continue
        cov = len(target & nc) / max(1, len(target))
        if cov < 0.7:
            continue
        t = (cov, int(node.get("depth", 0)), -len(nc))
        if t > best_t:
            best_t, best_id = t, str(nid)
    return best_id


def pick_tree_probe(
    candidates: Sequence[ScoredCase], mem: Memory,
) -> Probe | None:
    """从离线树中选最佳 probe。返回 None 表示树无法继续。"""
    tree = load_tree()
    if not tree or not candidates:
        return None

    cids = [c.case_id for c in candidates]
    nid = mem.tree_node or _locate(tree, cids)
    node = _get_node(tree, nid) or _get_node(tree, str(tree.get("root", "n")))
    if not node or node.get("is_leaf"):
        return None

    current = set(cids or node.get("case_ids", []))
    asked = set(mem.asked_probes)
    best_probe, best_s = None, -1.0

    for opt in node.get("probes", []):
        pid = str(opt.get("probe_id", ""))
        if not pid or pid in asked:
            continue
        pos = [c for c in opt.get("positive_ids", []) if c in current]
        neg = [c for c in opt.get("negative_ids", []) if c in current]
        unk = [c for c in opt.get("unknown_ids", []) if c in current]
        if not pos or not neg:
            continue
        local = split_quality(len(pos), len(neg), len(unk), len(current))
        combined = 0.6 * local + 0.4 * float(opt.get("score", 0) or 0)
        if combined <= best_s:
            continue
        best_s = combined
        best_probe = Probe(
            probe_id=pid, label=str(opt.get("label", "")),
            text=str(opt.get("question", "")),
            positive_ids=pos, negative_ids=neg, unknown_ids=unk,
            score=round(combined, 4), strategy="tree",
            tree_node=nid,
            yes_child=str(opt.get("yes_child") or node.get("yes_child") or ""),
            no_child=str(opt.get("no_child") or node.get("no_child") or ""),
            evidence=list(opt.get("evidence", []))[:5],
            debug={"node": nid, "global": opt.get("score", 0),
                   "local": round(local, 4), "pos": len(pos), "neg": len(neg)},
        )
    return best_probe
