"""CaseMemory 读写与跨轮更新。"""
from __future__ import annotations

from typing import Any

from medical_assistant.schemas import CaseMemory
from medical_assistant.session.answer_parser import parse_probe_answer


def load_memory(raw: dict[str, Any] | CaseMemory | None) -> CaseMemory:
    if isinstance(raw, CaseMemory):
        return raw
    return CaseMemory(**(raw or {})) if isinstance(raw, dict) else CaseMemory()


def dump_memory(mem: CaseMemory) -> dict[str, Any]:
    return mem.model_dump()


def _merge(existing: list[str], additions: list[str]) -> list[str]:
    out = list(existing or [])
    for item in additions:
        if item and item not in out:
            out.append(item)
    return out


def _save_probe_split(mem: CaseMemory, q: dict[str, Any]) -> None:
    fid = str(q.get("feature_id", ""))
    if not fid:
        return
    mem.probe_splits[fid] = {
        "positive": list(q.get("positive_case_ids", []) or []),
        "negative": list(q.get("negative_case_ids", []) or []),
        "unknown": list(q.get("unknown_case_ids", []) or []),
    }
    mem.probe_labels[fid] = str(q.get("label") or q.get("text") or fid)
    mem.probe_questions[fid] = str(q.get("text", ""))


def update_from_turn(
    raw: dict[str, Any] | None, *,
    question: str, query_en: str, key_terms: list[str], turn_index: int,
) -> CaseMemory:
    mem = load_memory(raw)
    is_answering = bool(mem.last_question_feature)

    if not mem.original_question:
        mem.original_question = question
        mem.normalized_query = query_en or question
    elif not is_answering:
        mem.normalized_query = query_en or question

    mem.key_terms = _merge(mem.key_terms, key_terms or [])
    mem.turn_index = turn_index
    mem.pending_answer_feature = ""
    mem.pending_answer_signal = ""

    if mem.last_question_feature:
        fid = mem.last_question_feature
        parsed = parse_probe_answer(
            question_text=mem.last_question_text, user_answer=question,
            probe_label=mem.probe_labels.get(fid, fid),
        )
        signal = parsed.signal
        if signal in {"yes", "no", "uncertain"}:
            mem.pending_answer_feature = fid
            mem.pending_answer_signal = signal
            if signal == "yes":
                mem.confirmed_features[fid] = {"value": True, "evidence": parsed.evidence}
                mem.denied_features.pop(fid, None)
                mem.uncertain_features.pop(fid, None)
                if mem.last_question_yes_child:
                    mem.tree_node_id = mem.last_question_yes_child
            elif signal == "no":
                mem.denied_features[fid] = {"value": True, "evidence": parsed.evidence}
                mem.confirmed_features.pop(fid, None)
                mem.uncertain_features.pop(fid, None)
                if mem.last_question_no_child:
                    mem.tree_node_id = mem.last_question_no_child
            else:
                mem.uncertain_features[fid] = {"answer": question, "evidence": parsed.evidence}
                if mem.last_question_tree_node_id:
                    mem.tree_node_id = mem.last_question_tree_node_id
        mem.key_terms = _merge(mem.key_terms, parsed.new_observations)
    return mem


def remember_question(raw: dict[str, Any] | CaseMemory | None, q: dict[str, Any]) -> CaseMemory:
    mem = load_memory(raw)
    qid = str(q.get("question_id", ""))
    fid = str(q.get("feature_id", ""))
    mem.last_question_id = qid
    mem.last_question_feature = fid
    mem.last_question_text = str(q.get("text", ""))
    mem.last_question_tree_node_id = str(q.get("tree_node_id", ""))
    mem.last_question_yes_child = str(q.get("yes_child_id", ""))
    mem.last_question_no_child = str(q.get("no_child_id", ""))
    if q.get("tree_node_id"):
        mem.tree_node_id = str(q["tree_node_id"])
    mem.asked_question_ids = _merge(mem.asked_question_ids, [qid])
    mem.asked_feature_ids = _merge(mem.asked_feature_ids, [fid])
    _save_probe_split(mem, q)
    return mem
