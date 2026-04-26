from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medical_assistant.config import get_settings
from medical_assistant.cases.store import load_cases
from medical_assistant.probes.anchor import mine_anchor_probes
from medical_assistant.probes.window import mine_window_probes, collect_text_units
from medical_assistant.probes.mine import mine_tree_probes
from medical_assistant.text.split import split_clauses
from medical_assistant.tree.builder import build_question_tree, save_question_tree, tree_stats


def print_tree(tree: dict, *, max_depth: int = 6) -> None:
    nodes = tree.get("nodes", {})

    def walk(nid: str, indent: str = "") -> None:
        node = nodes.get(nid, {})
        d = int(node.get("depth", 0))
        cc = len(node.get("case_ids", []))
        if d > max_depth:
            print(f"{indent}{nid} [...]"); return
        if node.get("is_leaf"):
            r = node.get("stop_reason", "")
            print(f"{indent}{nid} [leaf cases={cc} reason={r}]")
            br = node.get("best_rejected")
            if br:
                print(f"{indent}  best_rejected score={br.get('score')} label={br.get('label')}")
            return
        pid = node.get("primary_probe_id", "")
        opt = next((x for x in node.get("probe_options", []) if x.get("probe_id") == pid), None)
        if opt:
            dbg = opt.get("debug", {})
            print(f"{indent}{nid} [probe: {opt.get('label')} score={opt.get('split_score')} "
                  f"cases={cc} pos={dbg.get('positive','')} neg={dbg.get('negative','')} "
                  f"kind={dbg.get('kind','')}]")
        else:
            print(f"{indent}{nid} [internal cases={cc}]")
        if node.get("yes_child_id"):
            walk(node["yes_child_id"], indent + "  yes-> ")
        if node.get("no_child_id"):
            walk(node["no_child_id"], indent + "  no -> ")

    walk(tree.get("root_id", "n"))


def debug_probes(cases, limit: int = 12) -> None:
    print(f"\n{'='*60}\nDEBUG: {len(cases)} cases\n{'='*60}")
    for c in cases[:3]:
        cls = split_clauses(c.description)
        print(f"\n  [{c.case_id}] {c.title}  clauses={len(cls)}")
        for cl in cls[:4]:
            print(f"    - {cl}")
    units = collect_text_units(cases)
    print(f"\n  Text units: {len(units)}")
    anchors = mine_anchor_probes(cases, max_probes=limit, probe_prefix="dbg_a")
    print(f"\n  Anchor probes: {len(anchors)}")
    for p in anchors[:limit]:
        print(f"    score={p.split_score:.4f} pos={len(p.positive_case_ids):>2} "
              f"neg={len(p.negative_case_ids):>2} key={p.debug.get('primary_key','')} "
              f"label={p.label}")
    windows = mine_window_probes(cases, max_probes=limit, probe_prefix="dbg_w")
    print(f"\n  Window probes: {len(windows)}")
    for p in windows[:limit]:
        print(f"    score={p.split_score:.4f} pos={len(p.positive_case_ids):>2} "
              f"neg={len(p.negative_case_ids):>2} unk={len(p.unknown_case_ids):>2} "
              f"label={p.label}")
    merged = mine_tree_probes(cases, max_probes=limit, probe_prefix="dbg")
    print(f"\n  Merged: {len(merged)}")
    for p in merged[:limit]:
        print(f"    score={p.split_score:.4f} kind={p.debug.get('kind','')} label={p.label}")
    print(f"\n{'='*60}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-tree", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max-print-depth", type=int, default=6)
    args = parser.parse_args()

    cases = load_cases()
    print(f"Loaded cases: {len(cases)} from {get_settings().case_data_path}")
    if args.debug:
        debug_probes(cases)
    tree = build_question_tree(cases, debug=args.debug)
    path = save_question_tree(tree)
    print(f"Wrote question tree: {path}")
    print(json.dumps(tree_stats(tree), ensure_ascii=False, indent=2))
    if args.print_tree:
        print()
        print_tree(tree, max_depth=args.max_print_depth)


if __name__ == "__main__":
    main()
