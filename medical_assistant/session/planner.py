"""三阶段提问策略 + 终止判断。

阶段 1：沿离线决策树提问
阶段 2：在线从当前候选集动态挖 probe
阶段 3：候选集 ≤ 5 时逐例确认
"""
from __future__ import annotations

from medical_assistant.config import get_settings
from medical_assistant.schemas import CaseCandidate, CaseMemory, PlannedQuestion
from medical_assistant.text.split import split_observation_units
from medical_assistant.probes.mine import mine_tree_probes
from medical_assistant.tree.navigator import select_question_from_tree


def _asked(q: PlannedQuestion, s: set[str]) -> bool:
    return bool(q.feature_id and q.feature_id in s)


def _confirm_question(candidates: list[CaseCandidate], asked: set[str]) -> PlannedQuestion | None:
    for case in candidates[:3]:
        pid = f"confirm_{case.case_id}"
        if pid in asked:
            continue
        units = split_observation_units(case.title, case.description)
        evidence = units[:2] if units else [case.description[:80]]
        body = "；".join(x for x in evidence if x)
        text = (f'目前最接近"{case.title}"。想确认一下：{body}。'
                '这和你的情况相符吗？') if body else (
                f'目前最接近"{case.title}"。这和你的情况相符吗？')
        return PlannedQuestion(
            question_id=f"q_{pid}", feature_id=pid,
            label=f"确认：{case.title}", text=text,
            positive_case_ids=[case.case_id],
            negative_case_ids=[c.case_id for c in candidates if c.case_id != case.case_id],
            split_score=0.5, strategy="case_confirmation", evidence_texts=evidence,
            debug={"case_id": case.case_id},
        )
    return None


def select_question(
    candidates: list[CaseCandidate],
    memory: CaseMemory | None = None,
    asked_feature_ids: list[str] | None = None, **_,
) -> PlannedQuestion | None:
    if len(candidates) <= 1:
        return None
    settings = get_settings()
    mem = memory or CaseMemory(asked_feature_ids=asked_feature_ids or [])
    asked = set(asked_feature_ids or mem.asked_feature_ids or [])

    # 阶段 1
    tq = select_question_from_tree(candidates, mem)
    if tq and not _asked(tq, asked) and tq.split_score >= settings.tree_min_probe_gain:
        return tq

    # 阶段 2
    lp = mine_tree_probes(candidates, asked_probe_ids=asked, max_probes=1,
                          probe_prefix="online", min_child_size=1, min_child_ratio=0.0)
    if lp and lp[0].split_score >= settings.local_probe_min_gain:
        return lp[0].to_planned_question(strategy="online_dynamic_probe")

    # 阶段 3
    if len(candidates) <= 5:
        return _confirm_question(candidates, asked)
    return None


def should_finalize(
    candidates: list[CaseCandidate], turn_index: int,
    asked_feature_ids: list[str] | None = None,
    memory: CaseMemory | None = None,
) -> bool:
    if not candidates or len(candidates) <= 1:
        return True
    settings = get_settings()
    if turn_index >= settings.max_clarify_turns:
        return True

    if (turn_index >= settings.min_turns_before_finalize
            and len(candidates) <= settings.large_candidate_threshold
            and len(candidates) >= 2):
        gap = candidates[0].score - candidates[1].score
        if candidates[0].score >= 0.75 and gap >= settings.case_min_confidence_gap:
            return True

    return select_question(candidates, memory=memory, asked_feature_ids=asked_feature_ids) is None


def choose_final_case(candidates: list[CaseCandidate]) -> CaseCandidate | None:
    return candidates[0] if candidates else None
