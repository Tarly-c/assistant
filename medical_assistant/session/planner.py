"""三阶段提问策略 + 终止判断。

阶段 1（前期）：沿离线决策树提问
阶段 2（中期）：在线从当前候选集动态挖 probe
阶段 3（末期）：候选 ≤ 5 时逐例确认
"""
from __future__ import annotations

from medical_assistant.config import get_settings
from medical_assistant.schemas import Memory, Probe, ScoredCase
from medical_assistant.text.split import split_windows
from medical_assistant.probes.mine import mine_probes
from medical_assistant.tree.navigator import pick_tree_probe


def _confirm_probe(candidates: list[ScoredCase], asked: set[str]) -> Probe | None:
    """阶段 3：对评分最高的病例生成确认问题。"""
    for case in candidates[:3]:
        pid = f"confirm_{case.case_id}"
        if pid in asked:
            continue
        units = split_windows(case.title, case.description)
        body = "；".join(units[:2]) if units else case.description[:80]
        text = (f'目前最接近"{case.title}"。想确认一下：{body}。'
                f'这和你的情况相符吗？') if body else (
                f'目前最接近"{case.title}"。这和你的情况相符吗？')
        return Probe(
            probe_id=pid, label=f"确认：{case.title}", text=text,
            positive_ids=[case.case_id],
            negative_ids=[c.case_id for c in candidates if c.case_id != case.case_id],
            score=0.5, strategy="confirm",
            evidence=units[:2],
        )
    return None


def pick_probe(candidates: list[ScoredCase], mem: Memory) -> Probe | None:
    """选择下一个追问。按三阶段优先级。"""
    if len(candidates) <= 1:
        return None
    cfg = get_settings()
    asked = set(mem.asked_probes)

    # 阶段 1：离线树
    tp = pick_tree_probe(candidates, mem)
    if tp and tp.probe_id not in asked and tp.score >= cfg.tree_min_gain:
        return tp

    # 阶段 2：在线动态
    online = mine_probes(candidates, asked=asked, max_probes=1,
                         prefix="on", min_child=1)
    if online and online[0].score >= cfg.online_min_gain:
        return online[0].to_probe(strategy="online")

    # 阶段 3：逐例确认
    if len(candidates) <= 5:
        return _confirm_probe(candidates, asked)
    return None


def should_stop(candidates: list[ScoredCase], mem: Memory) -> bool:
    """判断是否应该终止追问，给出最终答案。"""
    if not candidates or len(candidates) <= 1:
        return True
    cfg = get_settings()
    if mem.turn >= cfg.max_turns:
        return True

    # 高置信度终止（需同时满足多个前提）
    if (mem.turn >= cfg.min_turns_to_finalize
            and len(candidates) <= cfg.large_set_threshold
            and candidates[0].score >= 0.75
            and candidates[0].score - candidates[1].score >= cfg.confidence_gap):
        return True

    return pick_probe(candidates, mem) is None
