from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medical_assistant.config import get_settings
from medical_assistant.cases.store import load_cases
from medical_assistant.probes.anchor import mine_anchor_probes
from medical_assistant.probes.window import mine_window_probes
from medical_assistant.probes.mine import mine_probes
from medical_assistant.text.split import split_clauses
from medical_assistant.tree.builder import build_tree, save_tree, tree_stats


def print_tree(tree: dict, max_depth: int = 6) -> None:
    nodes = tree.get("nodes", {})

    def walk(nid: str, indent: str = "") -> None:
        n = nodes.get(nid, {})
        d, cc = n.get("depth", 0), len(n.get("case_ids", []))
        if d > max_depth:
            print(f"{indent}{nid} [...]"); return
        if n.get("is_leaf"):
            print(f"{indent}{nid} [leaf cases={cc} reason={n.get('reason','')}]")
            if "rejected" in n:
                r = n["rejected"]
                print(f"{indent}  rejected: score={r.get('score')} label={r.get('label')}")
            return
        pid = n.get("primary", "")
        opt = next((x for x in n.get("probes", []) if x.get("probe_id") == pid), None)
        if opt:
            print(f"{indent}{nid} [probe: {opt.get('label')} score={opt.get('score')} "
                  f"cases={cc} pos={opt['debug'].get('pos','')} neg={opt['debug'].get('neg','')}]")
        else:
            print(f"{indent}{nid} [internal cases={cc}]")
        if n.get("yes_child"):
            walk(n["yes_child"], indent + "  yes-> ")
        if n.get("no_child"):
            walk(n["no_child"], indent + "  no -> ")

    walk(tree.get("root", "n"))


def debug_probes(cases, n: int = 12) -> None:
    print(f"\n{'='*60}\n{len(cases)} cases\n{'='*60}")
    for c in cases[:3]:
        cls = split_clauses(c.description)
        print(f"\n  [{c.case_id}] {c.title}  ({len(cls)} clauses)")
        for cl in cls[:4]:
            print(f"    - {cl}")
    a = mine_anchor_probes(cases, max_probes=n, prefix="dbg_a")
    print(f"\n  Anchors: {len(a)}")
    for p in a[:n]:
        print(f"    {p.score:.4f} pos={len(p.positive_ids):>2} neg={len(p.negative_ids):>2} {p.label}")
    w = mine_window_probes(cases, max_probes=n, prefix="dbg_w")
    print(f"\n  Windows: {len(w)}")
    for p in w[:n]:
        print(f"    {p.score:.4f} pos={len(p.positive_ids):>2} neg={len(p.negative_ids):>2} {p.label}")
    m = mine_probes(cases, max_probes=n, prefix="dbg")
    print(f"\n  Merged: {len(m)}")
    for p in m[:n]:
        print(f"    {p.score:.4f} {p.debug.get('kind','')} {p.label}")
    print(f"\n{'='*60}\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--print-tree", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--max-depth", type=int, default=6)
    args = p.parse_args()

    cases = load_cases()
    s = get_settings()
    print(f"Loaded {len(cases)} cases from {s.case_data_path}")
    if args.debug:
        debug_probes(cases)
    tree = build_tree(cases, debug=args.debug)
    path = save_tree(tree)
    print(f"Wrote: {path}")
    print(json.dumps(tree_stats(tree), indent=2))
    if args.print_tree:
        print()
        print_tree(tree, args.max_depth)


if __name__ == "__main__":
    main()
