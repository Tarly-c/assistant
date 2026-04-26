"""统一 Probe 挖掘入口：anchor + window 合并去重。"""
from __future__ import annotations

from typing import Iterable, Sequence

from medical_assistant.schemas import CaseRecord, ScoredCase
from medical_assistant.probes.types import ProbeCandidate
from medical_assistant.probes.anchor import mine_anchor_probes
from medical_assistant.probes.window import mine_window_probes


def mine_probes(
    cases: Sequence[CaseRecord | ScoredCase], *,
    asked: Iterable[str] = (), max_probes: int = 5,
    prefix: str = "p", min_child: int = 2,
) -> list[ProbeCandidate]:
    """合并 anchor + window 两路 probe，按 score 降序，positive set 去重。"""
    mc = max(1, min_child, int(len(cases) * 0.05))
    anchors = mine_anchor_probes(
        cases, asked=asked, max_probes=max_probes * 3,
        prefix=f"{prefix}_a", min_child=mc,
    )
    windows = mine_window_probes(
        cases, asked=asked, max_probes=max_probes * 2,
        prefix=f"{prefix}_w", min_child=mc,
    )
    merged, seen = [], set()
    for p in sorted([*anchors, *windows], key=lambda x: x.score, reverse=True):
        sig = frozenset(p.positive_ids)
        if sig not in seen:
            seen.add(sig)
            merged.append(p)
        if len(merged) >= max_probes:
            break
    return merged
