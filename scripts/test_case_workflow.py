from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run import run_one_turn


def run_dialog(name: str, messages: list[str]) -> None:
    print(f"\n=== {name} ===")
    sid = None
    last = None
    for msg in messages:
        last = run_one_turn(sid, msg)
        sid = last["session_id"]
        print("USER:", msg)
        print("BOT:", last["answer"].replace("\n", " | "))
        print("TYPE:", last["response_type"], "CANDIDATES:", last["candidate_count"])
    assert last is not None
    assert last["response_type"] in {"answer", "clarification"}
    assert last["candidate_count"] >= 1


def main() -> None:
    run_dialog(
        "wisdom tooth path",
        ["我最里面牙龈肿了，张嘴有点疼", "是", "是"],
    )
    run_dialog(
        "cold sweet path",
        ["喝冰水和吃甜的会酸一下，很快就好", "是", "不是"],
    )
    run_dialog(
        "crack bite path",
        ["咬坚果以后某颗后牙咬到某一下会电一下，松口也疼", "是", "不是"],
    )
    print("\nSmoke tests passed.")


if __name__ == "__main__":
    main()
