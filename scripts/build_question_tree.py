from __future__ import annotations

"""Build the offline case question tree.

Usage:
    python scripts/build_question_tree.py
    python scripts/build_question_tree.py --print-tree
    python scripts/build_question_tree.py --debug-root

The tree is generated from the configured case JSON using dynamic probe mining.
It does not use a hand-written disease/feature list and does not call an LLM.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from medical_assistant.config import get_settings
from medical_assistant.services.cases.features import collect_text_units, mine_local_probes
from medical_assistant.services.cases.question_tree import build_question_tree, save_question_tree, tree_stats
from medical_assistant.services.cases.store import load_cases


def _print_tree(tree: dict, *, max_nodes: int = 80) -> None:
    nodes = tree.get("nodes", {})
    root_id = tree.get("root_id", "n")

    def walk(node_id: str, prefix: str = "") -> None:
        nonlocal max_nodes
        if max_nodes <= 0:
            return
        node = nodes.get(node_id, {})
        max_nodes -= 1
        case_count = len(node.get("case_ids", []))
        if node.get("is_leaf"):
            print(f"{prefix}{node_id} [leaf cases={case_count} reason={node.get('stop_reason')}]")
            options = node.get("probe_options") or []
            if options:
                option = options[0]
                print(
                    f"{prefix}  best_rejected score={option.get('split_score')} "
                    f"label={option.get('label')} debug={option.get('debug')}"
                )
            return
        option = (node.get("probe_options") or [{}])[0]
        label = option.get("label") or option.get("prototype_text") or ""
        score = option.get("split_score", 0)
        debug = option.get("debug", {})
        print(f"{prefix}{node_id} [cases={case_count} score={score}] {label} debug={debug}")
        yes_child = node.get("yes_child_id")
        no_child = node.get("no_child_id")
        if yes_child:
            print(f"{prefix} ├─ 是 →", end=" ")
            walk(yes_child, prefix + " │ ")
        if no_child:
            print(f"{prefix} └─ 否 →", end=" ")
            walk(no_child, prefix + "   ")

    walk(root_id)


def _debug_root(cases) -> None:
    print("\n[debug-root] loaded cases:", len(cases))
    if cases:
        first = cases[0]
        print("[debug-root] first case:", first.case_id, first.title)
        print("[debug-root] first description length:", len(first.description))

    units = collect_text_units(cases)
    print("[debug-root] text units:", len(units))
    for unit in units[:12]:
        print(f"  - {unit.case_id}: {unit.text}")

    probes = mine_local_probes(
        cases,
        max_probes=12,
        min_score=0.0,
        probe_prefix="debug_root",
        min_child_size=2 if len(cases) >= 8 else 1,
        min_child_ratio=0.08 if len(cases) >= 20 else 0.0,
    )
    print("[debug-root] root probe candidates:", len(probes))
    for probe in probes:
        print(
            f"  - score={probe.split_score} label={probe.label} "
            f"pos={len(probe.positive_case_ids)} neg={len(probe.negative_case_ids)} "
            f"unk={len(probe.unknown_case_ids)} debug={probe.debug}"
        )
        if probe.evidence_texts:
            print("    evidence:", probe.evidence_texts[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-tree", action="store_true", help="print a compact tree preview")
    parser.add_argument("--debug-root", action="store_true", help="print root text units and candidate probes")
    parser.add_argument("--out", default="", help="optional output path; defaults to settings.case_question_tree_file")
    args = parser.parse_args()

    settings = get_settings()
    cases = load_cases()

    if args.debug_root:
        _debug_root(cases)

    tree = build_question_tree(cases)
    out = save_question_tree(tree, Path(args.out) if args.out else settings.case_question_tree_path)
    stats = tree_stats(tree)
    print(f"Loaded cases: {len(cases)} from {settings.case_data_path}")
    print(f"Wrote question tree: {out}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if args.print_tree:
        _print_tree(tree)


if __name__ == "__main__":
    main()
