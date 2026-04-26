from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from medical_assistant.services.cases.answer_parser import (
    build_answer_signal_messages,
    parse_answer_signal,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True, help="previous assistant question")
    parser.add_argument("--answer", required=True, help="current user reply")
    parser.add_argument("--label", default="", help="optional probe label")
    parser.add_argument("--invoke", action="store_true", help="actually invoke the configured LLM")
    args = parser.parse_args()

    messages = build_answer_signal_messages(
        args.answer,
        question_text=args.question,
        probe_label=args.label,
    )
    print(json.dumps({"messages": messages}, ensure_ascii=False, indent=2))

    if args.invoke:
        result = parse_answer_signal(
            args.answer,
            question_text=args.question,
            probe_label=args.label,
        )
        print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
