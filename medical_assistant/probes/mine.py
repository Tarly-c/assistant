"""统一 Probe 挖掘入口：合并 anchor + window 两路结果。"""
from __future__ import annotations

from typing import Iterable, Sequence

from medical_assistant.schemas import CaseCandidate, CaseRecord
from medical_assistant.probes.types import ProbeCandidate
from medical_assistant.probes.anchor import mine_anchor_probes
from medical_assistant.probes.window import mine_window_probes


def mine_tree_probes(
    cases: Sequence[CaseRecord | CaseCandidate], *,
    asked_probe_ids: Iterable[str] | None = None,
    max_probes: int = 5,
    probe_prefix: str = "probe",
    min_child_size: int = 2,
    min_child_ratio: float = 0.05,
) -> list[ProbeCandidate]:
    """返回 anchor + window 合并后的最佳 probes。"""
    min_cs = max(1, min_child_size, int(len(cases) * min_child_ratio))

    anchors = mine_anchor_probes(
        cases, asked_probe_ids=asked_probe_ids,
        max_probes=max_probes * 3, probe_prefix=f"{probe_prefix}_a",
        min_child_size=min_cs,
    )
    windows = mine_window_probes(
        cases, asked_probe_ids=asked_probe_ids,
        max_probes=max_probes * 2, probe_prefix=f"{probe_prefix}_w",
        min_child_size=min_cs,
    )

    merged: list[ProbeCandidate] = []
    seen: set[frozenset[str]] = set()
    for p in sorted([*anchors, *windows], key=lambda x: x.split_score, reverse=True):
        sig = frozenset(p.positive_case_ids)
        if sig not in seen:
            seen.add(sig)
            merged.append(p)
        if len(merged) >= max_probes:
            break
    return merged
