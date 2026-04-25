from __future__ import annotations

from medical_assistant.config import get_settings
from medical_assistant.schemas import CaseCandidate, PlannedQuestion
from medical_assistant.services.cases.features import all_features, get_feature
from medical_assistant.services.cases.store import entropy_split_score


def select_question(
    candidates: list[CaseCandidate],
    asked_feature_ids: list[str] | None = None,
    confirmed_feature_ids: list[str] | None = None,
    denied_feature_ids: list[str] | None = None,
) -> PlannedQuestion | None:
    """Pick the next feature question using the current candidate set.

    The score prefers features that split the remaining cases close to 50/50.
    It is deterministic and explainable, which is useful for the demo.
    """

    if len(candidates) <= 1:
        return None

    asked = set(asked_feature_ids or [])
    confirmed = set(confirmed_feature_ids or [])
    denied = set(denied_feature_ids or [])
    total = len(candidates)

    best: PlannedQuestion | None = None
    best_score = -1.0

    for feature in all_features():
        fid = feature.feature_id
        if fid in asked or fid in confirmed or fid in denied:
            continue

        positive = [c.case_id for c in candidates if fid in c.feature_tags]
        negative = [c.case_id for c in candidates if fid not in c.feature_tags]
        if not positive or not negative:
            continue

        split = entropy_split_score(len(positive), len(negative), total)
        # If a feature appears in high-ranked cases, prefer asking it a little.
        top_window = candidates[: min(8, len(candidates))]
        top_presence = sum(1 for c in top_window if fid in c.feature_tags) / max(1, len(top_window))
        score = split + 0.08 * top_presence

        if score > best_score:
            best_score = score
            best = PlannedQuestion(
                question_id=f"q_{fid}",
                feature_id=fid,
                text=feature.question,
                positive_case_ids=positive,
                negative_case_ids=negative,
                unknown_case_ids=[],
                split_score=round(score, 4),
            )

    return best


def should_finalize(
    candidates: list[CaseCandidate],
    turn_index: int,
    asked_feature_ids: list[str] | None = None,
) -> bool:
    if not candidates:
        return True
    if len(candidates) <= 1:
        return True

    settings = get_settings()
    if turn_index >= settings.max_clarify_turns:
        return True

    # Demo target: keep asking until the candidate set reaches one case,
    # unless we hit the turn limit or no useful feature remains.

    # If no useful feature remains, answer with the best ranked case.
    next_q = select_question(candidates, asked_feature_ids=asked_feature_ids)
    return next_q is None


def choose_final_case(candidates: list[CaseCandidate]) -> CaseCandidate | None:
    return candidates[0] if candidates else None
