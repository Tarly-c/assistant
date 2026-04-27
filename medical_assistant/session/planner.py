"""三阶段提问 + 终止判断。"""
from __future__ import annotations
from medical_assistant.config import get_settings
from medical_assistant.schemas import Memory, Probe, ScoredCase
from medical_assistant.text.split import split_windows
from medical_assistant.probes.mine import mine_probes
from medical_assistant.probes.scoring import rephrase
from medical_assistant.tree.navigator import pick_tree_probe


def _confirm_probe(candidates: list[ScoredCase], asked: set[str]) -> Probe | None:
    """阶段 3：逐例确认。"""
    for case in candidates[:3]:
        pid = f"confirm_{case.case_id}"
        if pid in asked:
            continue
        units = split_windows(case.title, case.description)
        body = "；".join(units[:2]) if units else case.description[:80]
        text = rephrase(f"确认是否符合：{case.title}", [body] if body else [])
        return Probe(
            probe_id=pid, label=f"确认：{case.title}", text=text,
            positive_ids=[case.case_id],
            negative_ids=[c.case_id for c in candidates if c.case_id != case.case_id],
            score=0.5, strategy="confirm",
        )
    return None


def pick_probe(candidates: list[ScoredCase], mem: Memory) -> Probe | None:
    if len(candidates) <= 1:
        return None
    cfg = get_settings()
    asked = set(mem.asked_probes)

    # 阶段 1：离线树
    tp = pick_tree_probe(candidates, mem)
    if tp and tp.probe_id not in asked and tp.score >= cfg.tree_min_gain:
        return tp

    # 阶段 2：在线动态（基于特征向量）
    # ★ 估算深度：用候选集大小作代理
    N = len(candidates)
    depth_hint = max(0, 4 - int(N / 15))  # N=60→0, N=30→2, N=5→4
    online = mine_probes(
        [c.case_id for c in candidates],
        asked=asked, max_probes=1, min_child=1, depth_hint=depth_hint,
    )
    if online and online[0].score >= cfg.online_min_gain:
        return online[0].to_probe(strategy="online")

    # 阶段 3：逐例确认
    if N <= 5:
        return _confirm_probe(candidates, asked)
    return None


def should_stop(candidates: list[ScoredCase], mem: Memory) -> bool:
    if not candidates or len(candidates) <= 1:
        return True
    cfg = get_settings()
    if mem.turn >= cfg.max_turns:
        return True
    if (mem.turn >= cfg.min_turns_to_finalize
            and len(candidates) <= cfg.large_set_threshold
            and candidates[0].score >= 0.75
            and candidates[0].score - candidates[1].score >= cfg.confidence_gap):
        return True
    return pick_probe(candidates, mem) is None
