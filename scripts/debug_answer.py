from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medical_assistant.session.parser import parse_answer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--probe", required=True, help="上一轮追问文本")
    p.add_argument("--input", required=True, help="用户回答")
    p.add_argument("--label", default="")
    args = p.parse_args()

    result = parse_answer(probe_text=args.probe, user_input=args.input,
                          probe_label=args.label)
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
