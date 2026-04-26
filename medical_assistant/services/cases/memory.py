from __future__ import annotations

from typing import Any

from medical_assistant.schemas import CaseMemory
from medical_assistant.services.cases.answer_parser import parse_probe_answer


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


def _remember_probe_split(memory: CaseMemory, question_dict: dict[str, Any]) -> None:
    fid = str(question_dict.get("feature_id", ""))
    if not fid:
        return
    memory.probe_splits[fid] = {
        "positive": list(question_dict.get("positive_case_ids", []) or []),
        "negative": list(question_dict.get("negative_case_ids", []) or []),
        "unknown": list(question_dict.get("unknown_case_ids", []) or []),
    }
    label = str(question_dict.get("label") or question_dict.get("text") or fid)
    memory.probe_labels[fid] = label
    memory.probe_questions[fid] = str(question_dict.get("text", ""))


def update_from_turn(
    raw_memory: dict[str, Any] | None,
    *,
    question: str,
    query_en: str,
    key_terms: list[str],
    turn_index: int,
) -> CaseMemory:
    """Update structured memory with the current user utterance.

    A user reply is interpreted relative to the previous assistant question via
    answer_parser. No yes/no word table is used here.
    """

    memory = load_memory(raw_memory)
    if not memory.original_question:
        memory.original_question = question
    memory.normalized_query = query_en or question
    memory.key_terms = merge_unique(memory.key_terms, key_terms or [])
    memory.turn_index = turn_index

    memory.pending_answer_feature = ""
    memory.pending_answer_signal = ""

    if memory.last_question_feature:
        fid = memory.last_question_feature
        parsed = parse_probe_answer(
            question_text=memory.last_question_text,
            user_answer=question,
            probe_label=memory.probe_labels.get(fid, fid),
            evidence_texts=[],
        )
        signal = parsed.signal
        if signal in {"yes", "no", "uncertain"}:
            memory.pending_answer_feature = fid
            memory.pending_answer_signal = signal
            if signal == "yes":
                memory.confirmed_features[fid] = {"value": True, "evidence": parsed.evidence}
                memory.denied_features.pop(fid, None)
                memory.uncertain_features.pop(fid, None)
                if memory.last_question_yes_child:
                    memory.tree_node_id = memory.last_question_yes_child
            elif signal == "no":
                memory.denied_features[fid] = {"value": True, "evidence": parsed.evidence}
                memory.confirmed_features.pop(fid, None)
                memory.uncertain_features.pop(fid, None)
                if memory.last_question_no_child:
                    memory.tree_node_id = memory.last_question_no_child
            else:
                memory.uncertain_features[fid] = {
                    "answer": question,
                    "evidence": parsed.evidence,
                }
                if memory.last_question_tree_node_id:
                    memory.tree_node_id = memory.last_question_tree_node_id

        # Extra observations are kept as raw key terms for subsequent lexical
        # ranking, without converting them into global static feature ids.
        memory.key_terms = merge_unique(memory.key_terms, parsed.new_observations)

    return memory


def remember_question(raw_memory: dict[str, Any] | CaseMemory | None, question_dict: dict[str, Any]) -> CaseMemory:
    memory = load_memory(raw_memory)
    qid = str(question_dict.get("question_id", ""))
    fid = str(question_dict.get("feature_id", ""))
    text = str(question_dict.get("text", ""))

    memory.last_question_id = qid
    memory.last_question_feature = fid
    memory.last_question_text = text
    memory.last_question_tree_node_id = str(question_dict.get("tree_node_id", ""))
    memory.last_question_yes_child = str(question_dict.get("yes_child_id", ""))
    memory.last_question_no_child = str(question_dict.get("no_child_id", ""))
    if question_dict.get("tree_node_id"):
        memory.tree_node_id = str(question_dict.get("tree_node_id"))

    memory.asked_question_ids = merge_unique(memory.asked_question_ids, [qid])
    memory.asked_feature_ids = merge_unique(memory.asked_feature_ids, [fid])
    _remember_probe_split(memory, question_dict)
    return memory


def clear_pending_answer(raw_memory: dict[str, Any] | None) -> CaseMemory:
    memory = load_memory(raw_memory)
    memory.pending_answer_feature = ""
    memory.pending_answer_signal = ""
    return memory
