from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medical_assistant.session.answer_parser import build_payload, parse_probe_answer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True)
    parser.add_argument("--answer", required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--invoke", action="store_true")
    args = parser.parse_args()

    payload = build_payload(question_text=args.question, user_answer=args.answer,
                            probe_label=args.label)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.invoke:
        result = parse_probe_answer(question_text=args.question, user_answer=args.answer,
                                    probe_label=args.label)
        print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
