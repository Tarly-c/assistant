from __future__ import annotations

from medical_assistant.config import get_settings
from medical_assistant.schemas import CaseCandidate, CaseMemory, PlannedQuestion
from medical_assistant.services.cases.features import mine_tree_probes, split_observation_units
from medical_assistant.services.cases.question_tree import select_question_from_tree


def _question_already_asked(question: PlannedQuestion, asked_feature_ids: set[str]) -> bool:
    return bool(question.feature_id and question.feature_id in asked_feature_ids)


def _plan_case_confirmation(
    candidates: list[CaseCandidate],
    asked_feature_ids: list[str] | None = None,
) -> PlannedQuestion | None:
    if len(candidates) <= 1:
        return None
    asked = set(asked_feature_ids or [])
    case = candidates[0]
    probe_id = f"confirm_{case.case_id}"
    if probe_id in asked:
        return None

    units = split_observation_units(case.title, case.description)
    evidence = units[:2] if units else [case.description[:80]]
    evidence_text = "；".join(x for x in evidence if x)
    if evidence_text:
        question_text = (
            f"目前最接近“{case.title}”。想确认一下：{evidence_text}。"
            "这和你的情况相符吗？请回答“是 / 不是 / 不确定”。"
        )
    else:
        question_text = (
            f"目前最接近“{case.title}”。这个判断和你的情况相符吗？"
            "请回答“是 / 不是 / 不确定”。"
        )

    return PlannedQuestion(
        question_id=f"q_{probe_id}",
        feature_id=probe_id,
        label=f"确认：{case.title}",
        text=question_text,
        positive_case_ids=[case.case_id],
        negative_case_ids=[c.case_id for c in candidates[1:]],
        unknown_case_ids=[],
        split_score=0.5,
        strategy="case_confirmation",
        evidence_texts=evidence,
        debug={"case_id": case.case_id, "candidate_count": len(candidates)},
    )


def select_question(
    candidates: list[CaseCandidate],
    memory: CaseMemory | None = None,
    asked_feature_ids: list[str] | None = None,
    confirmed_feature_ids: list[str] | None = None,
    denied_feature_ids: list[str] | None = None,
) -> PlannedQuestion | None:
    """Choose the next question.

    Priority:
    1. Follow the offline question tree.
    2. If the tree cannot split the current feasible set, mine a fresh broad/local
       probe from the current C.
    3. If C is small, ask case-level confirmation.
    """

    if len(candidates) <= 1:
        return None

    settings = get_settings()
    if memory is None:
        memory = CaseMemory(asked_feature_ids=asked_feature_ids or [])
    asked = set(asked_feature_ids or memory.asked_feature_ids or [])

    tree_question = select_question_from_tree(candidates, memory)
    if tree_question and not _question_already_asked(tree_question, asked):
        if tree_question.split_score >= settings.tree_min_probe_gain:
            return tree_question

    local_probes = mine_tree_probes(
        candidates,
        asked_probe_ids=asked,
        max_probes=1,
        probe_prefix="local",
        min_child_size=1,
        min_child_ratio=0.0,
    )
    if local_probes and local_probes[0].split_score >= settings.local_probe_min_gain:
        return local_probes[0].to_planned_question(strategy="local_dynamic_probe")

    if len(candidates) <= 5:
        return _plan_case_confirmation(candidates, asked_feature_ids=list(asked))
    return None


def should_finalize(
    candidates: list[CaseCandidate],
    turn_index: int,
    asked_feature_ids: list[str] | None = None,
    memory: CaseMemory | None = None,
) -> bool:
    if not candidates:
        return True
    if len(candidates) <= 1:
        return True

    settings = get_settings()
    if turn_index >= settings.max_clarify_turns:
        return True

    next_question = select_question(candidates, memory=memory, asked_feature_ids=asked_feature_ids)
    if next_question is None:
        return True

    if len(candidates) >= 2:
        gap = candidates[0].score - candidates[1].score
        if candidates[0].score >= 0.75 and gap >= settings.case_min_confidence_gap:
            return True
    return False


def choose_final_case(candidates: list[CaseCandidate]) -> CaseCandidate | None:
    return candidates[0] if candidates else None
