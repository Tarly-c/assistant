"""离线建树。"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medical_assistant.config import get_settings
from medical_assistant.cases.store import load_cases, load_clusters
from medical_assistant.tree.builder import build_tree, save_tree, tree_stats


def print_tree(tree: dict, max_depth: int = 6):
    nodes = tree.get("nodes", {})
    K = tree.get("K", 0)

    def walk(nid: str, indent: str = ""):
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
        probes = n.get("probes", [])
        if probes:
            p = probes[0]
            dim = p.get("feature_dim", -1)
            kind = "sem" if dim < K else "con"
            print(f"{indent}{nid} [dim={dim}({kind}) score={p.get('score')} "
                  f"cases={cc} label={p.get('label','')}]")
        else:
            print(f"{indent}{nid} [internal cases={cc}]")
        if n.get("yes_child"):
            walk(n["yes_child"], indent + "  yes→ ")
        if n.get("no_child"):
            walk(n["no_child"], indent + "  no → ")

    walk(tree.get("root", "n"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--print-tree", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--max-depth", type=int, default=6)
    args = p.parse_args()

    cases = load_cases()
    cfg = get_settings()
    print(f"Loaded {len(cases)} cases")

    tree = build_tree(cases, debug=args.debug)
    path = save_tree(tree)
    print(f"Wrote tree: {path}")
    print(json.dumps(tree_stats(tree), indent=2))

    if args.print_tree:
        print()
        print_tree(tree, args.max_depth)


if __name__ == "__main__":
    main()
