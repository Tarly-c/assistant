from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from medical_assistant.services.cases.answer_parser import build_probe_answer_payload, parse_probe_answer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True, help="上一轮 assistant 问题")
    parser.add_argument("--answer", required=True, help="用户本轮回答")
    parser.add_argument("--label", default="", help="probe label")
    parser.add_argument("--invoke", action="store_true", help="显式调用 LLM；默认只打印 payload")
    args = parser.parse_args()

    payload = build_probe_answer_payload(
        question_text=args.question,
        user_answer=args.answer,
        probe_label=args.label,
        evidence_texts=[],
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.invoke:
        result = parse_probe_answer(
            question_text=args.question,
            user_answer=args.answer,
            probe_label=args.label,
            evidence_texts=[],
        )
        print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
