from __future__ import annotations

"""Build the offline case question tree.

Usage:
    python scripts/build_question_tree.py
    python scripts/build_question_tree.py --print-tree

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
from medical_assistant.services.cases.question_tree import build_question_tree, save_question_tree, tree_stats
from medical_assistant.services.cases.store import load_cases


def _print_tree(tree: dict, *, max_nodes: int = 60) -> None:
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
            return
        option = (node.get("probe_options") or [{}])[0]
        label = option.get("label") or option.get("prototype_text") or "<no probe>"
        score = option.get("split_score", 0)
        print(f"{prefix}{node_id} [cases={case_count} score={score}] {label}")
        yes_child = node.get("yes_child_id")
        no_child = node.get("no_child_id")
        if yes_child:
            print(f"{prefix}  ├─ 是 →", end=" ")
            walk(yes_child, prefix + "  │  ")
        if no_child:
            print(f"{prefix}  └─ 否 →", end=" ")
            walk(no_child, prefix + "     ")

    walk(root_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-tree", action="store_true", help="print a compact tree preview")
    parser.add_argument("--out", default="", help="optional output path; defaults to settings.case_question_tree_file")
    args = parser.parse_args()

    settings = get_settings()
    cases = load_cases()
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
