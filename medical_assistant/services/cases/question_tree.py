from __future__ import annotations

"""Offline/online question-tree support.

A tree node represents a current feasible case subset.  Its primary probe is
mined from that subset, then yes/no children are recursively built from the
corresponding split.  Online planning follows the tree when possible and falls
back to local dynamic probe mining when the tree has no useful split.
"""

from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from medical_assistant.config import get_settings
from medical_assistant.schemas import CaseCandidate, CaseMemory, CaseRecord, PlannedQuestion
from medical_assistant.services.cases.features import ProbeCandidate, mine_local_probes, split_quality

TREE_VERSION = 1


def _unique(ids: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for cid in ids:
        if cid and cid not in seen:
            out.append(cid)
            seen.add(cid)
    return out


def _probe_to_dict(probe: ProbeCandidate) -> dict[str, Any]:
    return {
        "probe_id": probe.probe_id,
        "label": probe.label,
        "prototype_text": probe.prototype_text,
        "question_seed": probe.question_seed,
        "positive_case_ids": probe.positive_case_ids,
        "negative_case_ids": probe.negative_case_ids,
        "unknown_case_ids": probe.unknown_case_ids,
        "evidence_texts": probe.evidence_texts,
        "split_score": probe.split_score,
        "debug": probe.debug,
        "yes_child_id": "",
        "no_child_id": "",
    }


def _branch_ids(probe_dict: dict[str, Any], branch: str, *, include_unknown: bool) -> list[str]:
    if branch == "yes":
        ids = list(probe_dict.get("positive_case_ids", []))
    else:
        ids = list(probe_dict.get("negative_case_ids", []))
    if include_unknown:
        ids.extend(probe_dict.get("unknown_case_ids", []))
    return _unique(ids)


def build_question_tree(cases: Sequence[CaseRecord]) -> dict[str, Any]:
    """Build a deterministic, dependency-free question tree from case texts."""

    settings = get_settings()
    case_map = {case.case_id: case for case in cases}
    nodes: dict[str, dict[str, Any]] = {}

    def build_node(case_ids: list[str], depth: int, node_id: str, asked_probe_ids: list[str]) -> str:
        case_ids = _unique(cid for cid in case_ids if cid in case_map)
        node: dict[str, Any] = {
            "node_id": node_id,
            "depth": depth,
            "case_ids": case_ids,
            "is_leaf": False,
            "stop_reason": "",
            "probe_options": [],
            "primary_probe_id": "",
            "yes_child_id": "",
            "no_child_id": "",
            "leaf_case_ids": [],
        }
        nodes[node_id] = node

        if not case_ids:
            node["is_leaf"] = True
            node["stop_reason"] = "empty_case_set"
            return node_id
        if depth >= settings.tree_max_depth:
            node["is_leaf"] = True
            node["stop_reason"] = "max_depth"
            node["leaf_case_ids"] = case_ids
            return node_id
        if len(case_ids) <= settings.tree_min_leaf_cases:
            node["is_leaf"] = True
            node["stop_reason"] = "small_leaf"
            node["leaf_case_ids"] = case_ids
            return node_id

        subset = [case_map[cid] for cid in case_ids]
        probes = mine_local_probes(
            subset,
            asked_probe_ids=asked_probe_ids,
            max_probes=max(1, settings.tree_probe_options_per_node),
            min_score=settings.tree_min_probe_gain,
            probe_prefix=f"{node_id}_p",
            min_child_size=2 if len(case_ids) >= 8 else 1,
            min_child_ratio=0.12 if len(case_ids) >= 8 else 0.0,
        )
        if not probes:
            node["is_leaf"] = True
            node["stop_reason"] = "no_probe"
            node["leaf_case_ids"] = case_ids
            return node_id

        probe_dicts = [_probe_to_dict(probe) for probe in probes]
        primary = probe_dicts[0]
        if float(primary.get("split_score", 0.0)) < settings.tree_min_probe_gain:
            node["is_leaf"] = True
            node["stop_reason"] = "low_gain"
            node["leaf_case_ids"] = case_ids
            return node_id

        yes_ids = _branch_ids(primary, "yes", include_unknown=settings.tree_use_unknown_as_soft_branch)
        no_ids = _branch_ids(primary, "no", include_unknown=settings.tree_use_unknown_as_soft_branch)
        # If a soft branch does not shrink at all, remove unknowns for that branch.
        if set(yes_ids) == set(case_ids):
            yes_ids = _branch_ids(primary, "yes", include_unknown=False)
        if set(no_ids) == set(case_ids):
            no_ids = _branch_ids(primary, "no", include_unknown=False)

        if not yes_ids or not no_ids:
            node["is_leaf"] = True
            node["stop_reason"] = "degenerate_split"
            node["leaf_case_ids"] = case_ids
            return node_id

        node["probe_options"] = probe_dicts
        node["primary_probe_id"] = primary["probe_id"]

        yes_child = f"{node_id}_y"
        no_child = f"{node_id}_n"
        node["yes_child_id"] = yes_child
        node["no_child_id"] = no_child
        primary["yes_child_id"] = yes_child
        primary["no_child_id"] = no_child

        next_asked = asked_probe_ids + [str(primary["probe_id"])]
        build_node(yes_ids, depth + 1, yes_child, next_asked)
        build_node(no_ids, depth + 1, no_child, next_asked)
        return node_id

    root_id = "n"
    build_node([case.case_id for case in cases], 0, root_id, [])
    return {
        "version": TREE_VERSION,
        "root_id": root_id,
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
    settings = get_settings()
    path = path or settings.case_question_tree_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tree, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


@lru_cache(maxsize=1)
def load_question_tree() -> dict[str, Any] | None:
    path = get_settings().case_question_tree_path
    if not path.exists():
        return None
    try:
        tree = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(tree, dict) or "nodes" not in tree:
        return None
    return tree


def clear_question_tree_cache() -> None:
    load_question_tree.cache_clear()


def _candidate_ids_for_planning(candidates: Sequence[CaseCandidate]) -> list[str]:
    if not candidates:
        return []
    # If the current utterance strongly matched a subset, use that subset for
    # planning; otherwise use the whole feasible set.  This prevents the tree
    # from ignoring information already present in the user's first message.
    scored = [c.case_id for c in candidates if c.score >= 0.12]
    if 3 <= len(scored) <= max(3, len(candidates) * 0.75):
        return scored
    return [c.case_id for c in candidates]


def _node(tree: dict[str, Any], node_id: str) -> dict[str, Any] | None:
    nodes = tree.get("nodes", {})
    if isinstance(nodes, dict):
        item = nodes.get(node_id)
        if isinstance(item, dict):
            return item
    return None


def _best_node_for_candidates(tree: dict[str, Any], candidate_ids: list[str]) -> str:
    root_id = str(tree.get("root_id", "n"))
    if not candidate_ids:
        return root_id
    target = set(candidate_ids)
    best_id = root_id
    best_tuple = (-1.0, -1, 10**9)
    for node_id, node in tree.get("nodes", {}).items():
        node_case_ids = set(node.get("case_ids", []))
        if not node_case_ids:
            continue
        overlap = len(target & node_case_ids)
        if overlap == 0:
            continue
        coverage = overlap / max(1, len(target))
        if coverage < 0.7:
            continue
        # Prefer high coverage, deeper/smaller nodes.
        score_tuple = (coverage, int(node.get("depth", 0)), -len(node_case_ids))
        if score_tuple > best_tuple:
            best_tuple = score_tuple
            best_id = str(node_id)
    return best_id


def select_question_from_tree(
    candidates: Sequence[CaseCandidate],
    memory: CaseMemory,
) -> PlannedQuestion | None:
    tree = load_question_tree()
    if not tree or not candidates:
        return None

    candidate_ids = _candidate_ids_for_planning(candidates)
    if memory.tree_node_id:
        node_id = memory.tree_node_id
    else:
        node_id = _best_node_for_candidates(tree, candidate_ids)
    node = _node(tree, node_id) or _node(tree, str(tree.get("root_id", "n")))
    if not node or node.get("is_leaf"):
        return None

    current_set = set(candidate_ids or node.get("case_ids", []))
    asked = set(memory.asked_feature_ids or [])
    best_question: PlannedQuestion | None = None
    best_score = -1.0

    for option in node.get("probe_options", []):
        if not isinstance(option, dict):
            continue
        probe_id = str(option.get("probe_id", ""))
        if not probe_id or probe_id in asked:
            continue
        positive = [cid for cid in option.get("positive_case_ids", []) if cid in current_set]
        negative = [cid for cid in option.get("negative_case_ids", []) if cid in current_set]
        unknown = [cid for cid in option.get("unknown_case_ids", []) if cid in current_set]
        if not positive or not negative:
            continue
        local_score = split_quality(len(positive), len(negative), len(unknown), len(current_set))
        combined_score = 0.65 * local_score + 0.35 * float(option.get("split_score", 0.0) or 0.0)
        if combined_score <= best_score:
            continue
        best_score = combined_score
        best_question = PlannedQuestion(
            question_id=f"q_{node_id}_{probe_id}",
            feature_id=probe_id,
            label=str(option.get("label", "")),
            text=str(option.get("question_seed", "")),
            positive_case_ids=positive,
            negative_case_ids=negative,
            unknown_case_ids=unknown,
            split_score=round(combined_score, 4),
            strategy="tree_probe",
            tree_node_id=node_id,
            yes_child_id=str(option.get("yes_child_id") or node.get("yes_child_id") or ""),
            no_child_id=str(option.get("no_child_id") or node.get("no_child_id") or ""),
            evidence_texts=list(option.get("evidence_texts", []))[:5],
            debug={
                "tree_node_id": node_id,
                "global_split_score": option.get("split_score", 0.0),
                "local_split_score": round(local_score, 4),
                "positive": len(positive),
                "negative": len(negative),
                "unknown": len(unknown),
            },
        )
    return best_question


def tree_stats(tree: dict[str, Any]) -> dict[str, int]:
    nodes = tree.get("nodes", {}) if isinstance(tree, dict) else {}
    leaf_count = 0
    internal_count = 0
    max_depth = 0
    for node in nodes.values():
        if not isinstance(node, dict):
            continue
        max_depth = max(max_depth, int(node.get("depth", 0)))
        if node.get("is_leaf"):
            leaf_count += 1
        else:
            internal_count += 1
    return {"nodes": len(nodes), "internal": internal_count, "leaves": leaf_count, "max_depth": max_depth}
