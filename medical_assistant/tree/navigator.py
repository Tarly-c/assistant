"""在线沿离线决策树选择提问。"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Sequence

from medical_assistant.config import get_settings
from medical_assistant.schemas import CaseCandidate, CaseMemory, PlannedQuestion
from medical_assistant.probes.scoring import split_quality


@lru_cache(maxsize=1)
def load_question_tree() -> dict[str, Any] | None:
    path = get_settings().case_question_tree_path
    if not path.exists():
        return None
    try:
        tree = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return tree if isinstance(tree, dict) and "nodes" in tree else None


def clear_tree_cache() -> None:
    load_question_tree.cache_clear()


def _node(tree: dict, nid: str) -> dict[str, Any] | None:
    item = tree.get("nodes", {}).get(nid)
    return item if isinstance(item, dict) else None


def _best_node_for(tree: dict, cids: list[str]) -> str:
    root = str(tree.get("root_id", "n"))
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


def select_question_from_tree(
    candidates: Sequence[CaseCandidate], memory: CaseMemory,
) -> PlannedQuestion | None:
    tree = load_question_tree()
    if not tree or not candidates:
        return None

    cids = [c.case_id for c in candidates]
    nid = memory.tree_node_id or _best_node_for(tree, cids)
    node = _node(tree, nid) or _node(tree, str(tree.get("root_id", "n")))
    if not node or node.get("is_leaf"):
        return None

    current = set(cids or node.get("case_ids", []))
    asked = set(memory.asked_feature_ids or [])
    best_q, best_s = None, -1.0

    for opt in node.get("probe_options", []):
        if not isinstance(opt, dict):
            continue
        pid = str(opt.get("probe_id", ""))
        if not pid or pid in asked:
            continue
        pos = [c for c in opt.get("positive_case_ids", []) if c in current]
        neg = [c for c in opt.get("negative_case_ids", []) if c in current]
        unk = [c for c in opt.get("unknown_case_ids", []) if c in current]
        if not pos or not neg:
            continue
        local = split_quality(len(pos), len(neg), len(unk), len(current))
        combined = 0.6 * local + 0.4 * float(opt.get("split_score", 0) or 0)
        if combined <= best_s:
            continue
        best_s = combined
        best_q = PlannedQuestion(
            question_id=f"q_{nid}_{pid}", feature_id=pid,
            label=str(opt.get("label", "")),
            text=str(opt.get("question_seed", "")),
            positive_case_ids=pos, negative_case_ids=neg, unknown_case_ids=unk,
            split_score=round(combined, 4), strategy="tree_probe",
            tree_node_id=nid,
            yes_child_id=str(opt.get("yes_child_id") or node.get("yes_child_id") or ""),
            no_child_id=str(opt.get("no_child_id") or node.get("no_child_id") or ""),
            evidence_texts=list(opt.get("evidence_texts", []))[:5],
            debug={"tree_node_id": nid, "global_score": opt.get("split_score", 0),
                   "local_score": round(local, 4),
                   "positive": len(pos), "negative": len(neg), "unknown": len(unk)},
        )
    return best_q
