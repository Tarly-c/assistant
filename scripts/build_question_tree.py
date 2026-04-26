from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from medical_assistant.services.cases.features import collect_text_units, mine_tree_probes
from medical_assistant.services.cases.question_tree import build_question_tree, save_question_tree, tree_stats
from medical_assistant.services.cases.store import load_cases


def print_tree(tree: dict, *, max_depth: int = 4) -> None:
    nodes = tree.get("nodes", {})
    root_id = tree.get("root_id", "n")

    def walk(node_id: str, indent: str = "") -> None:
        node = nodes.get(node_id, {})
        depth = int(node.get("depth", 0))
        if depth > max_depth:
            print(f"{indent}{node_id} [...]")
            return
        case_count = len(node.get("case_ids", []))
        if node.get("is_leaf"):
            reason = node.get("stop_reason", "")
            print(f"{indent}{node_id} [leaf cases={case_count} reason={reason}]")
            if node.get("best_rejected"):
                br = node["best_rejected"]
                print(
                    f"{indent}  best_rejected score={br.get('score')} "
                    f"label={br.get('label')} debug={br.get('debug')}"
                )
            return
        primary = node.get("primary_probe_id", "")
        option = next((x for x in node.get("probe_options", []) if x.get("probe_id") == primary), None)
        if option:
            print(
                f"{indent}{node_id} [probe={option.get('label')} "
                f"score={option.get('split_score')} cases={case_count} "
                f"debug={option.get('debug')} ]"
            )
        else:
            print(f"{indent}{node_id} [internal cases={case_count}]")
        if node.get("yes_child_id"):
            walk(node["yes_child_id"], indent + "  yes-> ")
        if node.get("no_child_id"):
            walk(node["no_child_id"], indent + "  no -> ")

    walk(root_id)


def debug_root(cases, limit: int = 12) -> None:
    units = collect_text_units(cases)
    print(f"DEBUG text_units={len(units)}")
    for unit in units[:5]:
        print(f"  unit {unit.case_id}: {unit.text[:120]}")
    probes = mine_tree_probes(
        cases,
        asked_probe_ids=[],
        max_probes=limit,
        probe_prefix="debug_root",
        min_child_size=2 if len(cases) >= 8 else 1,
        min_child_ratio=0.08 if len(cases) >= 8 else 0.0,
    )
    print(f"DEBUG root_probes={len(probes)}")
    for probe in probes[:limit]:
        print(
            json.dumps(
                {
                    "label": probe.label,
                    "score": probe.split_score,
                    "pos": len(probe.positive_case_ids),
                    "neg": len(probe.negative_case_ids),
                    "unknown": len(probe.unknown_case_ids),
                    "evidence": probe.evidence_texts[:2],
                    "debug": probe.debug,
                },
                ensure_ascii=False,
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-tree", action="store_true")
    parser.add_argument("--debug-root", action="store_true")
    parser.add_argument("--max-print-depth", type=int, default=4)
    args = parser.parse_args()

    cases = load_cases()
    print(f"Loaded cases: {len(cases)}")
    if args.debug_root:
        debug_root(cases)

    tree = build_question_tree(cases)
    path = save_question_tree(tree)
    print(f"Wrote question tree: {path}")
    print(json.dumps(tree_stats(tree), ensure_ascii=False, indent=2))
    if args.print_tree:
        print_tree(tree, max_depth=args.max_print_depth)


if __name__ == "__main__":
    main()
