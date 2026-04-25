from __future__ import annotations

from typing import Any

from medical_assistant.schemas import CaseMemory
from medical_assistant.services.cases.features import classify_answer, extract_features


def load_memory(raw: dict[str, Any] | CaseMemory | None) -> CaseMemory:
    if isinstance(raw, CaseMemory):
        return raw
    if isinstance(raw, dict):
        return CaseMemory(**raw)
    return CaseMemory()


def dump_memory(memory: CaseMemory) -> dict[str, Any]:
    return memory.model_dump()


def merge_unique(existing: list[str], additions: list[str]) -> list[str]:
    out = list(existing or [])
    for item in additions:
        if item and item not in out:
            out.append(item)
    return out


def update_from_turn(
    raw_memory: dict[str, Any] | None,
    *,
    question: str,
    query_en: str,
    key_terms: list[str],
    turn_index: int,
) -> CaseMemory:
    """Update structured memory with the current user utterance.

    The previous assistant question is stored as last_question_feature. When the
    user replies "是/不是/不确定", we bind that answer to the feature instead of
    asking the LLM to reconstruct context from chat history.
    """

    memory = load_memory(raw_memory)
    if not memory.original_question:
        memory.original_question = question
    memory.normalized_query = query_en or question
    memory.key_terms = merge_unique(memory.key_terms, key_terms or [])
    memory.turn_index = turn_index

    # Parse answer to the previous planned feature question.
    memory.pending_answer_feature = ""
    memory.pending_answer_signal = ""
    if memory.last_question_feature:
        signal = classify_answer(question, memory.last_question_feature)
        if signal in {"yes", "no", "uncertain"}:
            fid = memory.last_question_feature
            memory.pending_answer_feature = fid
            memory.pending_answer_signal = signal
            if signal == "yes":
                memory.confirmed_features[fid] = True
                memory.denied_features.pop(fid, None)
                memory.uncertain_features.pop(fid, None)
            elif signal == "no":
                memory.denied_features[fid] = True
                memory.confirmed_features.pop(fid, None)
                memory.uncertain_features.pop(fid, None)
            else:
                memory.uncertain_features[fid] = question

    # Also extract explicit features from the text. This helps answers like
    # "不是智齿，但喝冷水会酸" move the workflow forward.
    for fid in extract_features(question):
        if fid not in memory.denied_features:
            memory.confirmed_features.setdefault(fid, True)

    return memory


def remember_question(raw_memory: dict[str, Any] | None, question_dict: dict[str, Any]) -> CaseMemory:
    memory = load_memory(raw_memory)
    qid = str(question_dict.get("question_id", ""))
    fid = str(question_dict.get("feature_id", ""))
    text = str(question_dict.get("text", ""))

    memory.last_question_id = qid
    memory.last_question_feature = fid
    memory.last_question_text = text
    memory.asked_question_ids = merge_unique(memory.asked_question_ids, [qid])
    memory.asked_feature_ids = merge_unique(memory.asked_feature_ids, [fid])
    return memory


def clear_pending_answer(raw_memory: dict[str, Any] | None) -> CaseMemory:
    memory = load_memory(raw_memory)
    memory.pending_answer_feature = ""
    memory.pending_answer_signal = ""
    return memory
