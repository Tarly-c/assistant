from __future__ import annotations
import argparse, json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medical_assistant.session.parser import parse_answer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--probe", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--label", default="")
    args = p.parse_args()
    r = parse_answer(probe_text=args.probe, user_input=args.input, probe_label=args.label)
    print(json.dumps(r.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
