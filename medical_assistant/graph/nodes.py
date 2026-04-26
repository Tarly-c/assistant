"""所有 workflow 节点函数。"""
from __future__ import annotations

from medical_assistant.config import get_settings
from medical_assistant.graph.state import GraphState
from medical_assistant.llm import invoke_structured
from medical_assistant.prompts import NORMALIZE_PROMPT
from medical_assistant.schemas import CaseCandidate, CaseMemory, NormalizedInput
from medical_assistant.text.split import extract_search_terms
from medical_assistant.cases.store import all_case_ids
from medical_assistant.cases.ranker import (
    apply_feature_filters, confidence_from_candidates,
    search_cases, top_candidate_summary,
)
from medical_assistant.session.memory import dump_memory, load_memory, remember_question
from medical_assistant.session.planner import choose_final_case, select_question, should_finalize
from medical_assistant.tree.navigator import load_question_tree


# ── normalize ──

def normalize_node(state: GraphState) -> dict:
    question = state.get("question", "")
    settings = get_settings()
    mem_raw = state.get("case_memory") or {}
    is_answering = bool(mem_raw.get("last_question_feature", ""))

    if is_answering:
        prev_q = mem_raw.get("original_question") or mem_raw.get("normalized_query") or question
        prev_terms = list(mem_raw.get("key_terms") or [])
        for t in extract_search_terms(question):
            if len(t) >= 3 and t not in prev_terms:
                prev_terms.append(t)
        return {"query_en": prev_q, "intent": state.get("intent") or "general",
                "key_terms_en": prev_terms[:15]}

    result = NormalizedInput()
    if settings.use_llm_normalize:
        result = invoke_structured(NormalizedInput, [
            {"role": "system", "content": NORMALIZE_PROMPT},
            {"role": "user", "content": f"当前问题: {question}"},
        ])
    terms: list[str] = []
    for t in (result.key_terms_en or []) + extract_search_terms(question):
        if t and t not in terms:
            terms.append(t)
    return {"query_en": result.query_en or question, "intent": result.intent or "general",
            "key_terms_en": terms[:10]}


# ── update_memory ──

def update_memory_node(state: GraphState) -> dict:
    from medical_assistant.session.memory import update_from_turn
    mem = update_from_turn(
        state.get("case_memory"), question=state.get("question", ""),
        query_en=state.get("query_en", ""), key_terms=state.get("key_terms_en", []),
        turn_index=state.get("turn_index", 0),
    )
    return {"case_memory": dump_memory(mem),
            "confirmed_features": mem.confirmed_features,
            "denied_features": mem.denied_features,
            "uncertain_features": mem.uncertain_features,
            "tree_node_id": mem.tree_node_id}


# ── narrow_cases ──

def _tree_node_ids(nid: str) -> list[str]:
    if not nid:
        return []
    tree = load_question_tree()
    if not tree:
        return []
    node = (tree.get("nodes") or {}).get(nid)
    return list(node.get("case_ids", [])) if isinstance(node, dict) else []


def narrow_cases_node(state: GraphState) -> dict:
    mem = load_memory(state.get("case_memory"))
    tree_ids = _tree_node_ids(mem.tree_node_id)
    base = tree_ids or mem.candidate_case_ids or all_case_ids()
    narrowed = apply_feature_filters(
        base, confirmed_features=mem.confirmed_features.keys(),
        denied_features=mem.denied_features.keys(), probe_splits=mem.probe_splits,
    ) or base
    mem.candidate_case_ids = narrowed
    mem.pending_answer_feature = ""
    mem.pending_answer_signal = ""
    return {"candidate_case_ids": narrowed, "candidate_count": len(narrowed),
            "tree_node_id": mem.tree_node_id, "case_memory": dump_memory(mem)}


# ── retrieve_cases ──

def retrieve_cases_node(state: GraphState) -> dict:
    mem = load_memory(state.get("case_memory"))
    query = mem.original_question or state.get("query_en") or state.get("question", "")
    terms = mem.key_terms or state.get("key_terms_en", [])
    candidates = search_cases(
        query=query, terms=terms,
        candidate_ids=state.get("candidate_case_ids") or mem.candidate_case_ids or None,
        confirmed_features=mem.confirmed_features.keys(),
        denied_features=mem.denied_features.keys(),
        probe_splits=mem.probe_splits, top_k=0,
    )
    mem.candidate_case_ids = [c.case_id for c in candidates]
    bs = candidates[0].score if candidates else 0.0
    conf = confidence_from_candidates(candidates)
    return {
        "case_candidates": [c.model_dump() for c in candidates],
        "candidate_case_ids": mem.candidate_case_ids,
        "candidate_scores": {c.case_id: c.score for c in candidates},
        "candidate_count": len(candidates),
        "top_candidates": top_candidate_summary(candidates),
        "hits": [{"case_id": c.case_id, "title": c.title, "snippet": c.description[:240],
                  "score": c.score} for c in candidates[:5]],
        "best_score": bs, "confidence": conf, "enough": bool(candidates),
        "case_memory": dump_memory(mem),
    }


# ── route ──

def _cands(state: GraphState) -> list[CaseCandidate]:
    return [CaseCandidate(**x) for x in state.get("case_candidates", [])]


def route_after_cases(state: GraphState) -> str:
    mem = load_memory(state.get("case_memory"))
    if should_finalize(_cands(state), state.get("turn_index", 0),
                       asked_feature_ids=mem.asked_feature_ids, memory=mem):
        return "final_answer"
    return "plan_question"


# ── plan_question ──

def plan_question_node(state: GraphState) -> dict:
    mem = load_memory(state.get("case_memory"))
    candidates = _cands(state)
    q = select_question(candidates, memory=mem, asked_feature_ids=mem.asked_feature_ids,
                        confirmed_feature_ids=list(mem.confirmed_features),
                        denied_feature_ids=list(mem.denied_features))
    if q is None:
        return {"should_answer": True, "response_type": "answer", "phase": "ANSWERED"}
    mem = remember_question(mem, q.model_dump())
    return {
        "response_type": "clarification", "answer": q.text,
        "selected_question": q.model_dump(), "case_memory": dump_memory(mem),
        "confidence": state.get("confidence", 0.0), "phase": "NEEDS_CLARIFICATION",
        "turn_index": state.get("turn_index", 0) + 1,
        "sources": [c.title for c in candidates[:3]],
    }


# ── final_answer ──

def final_answer_node(state: GraphState) -> dict:
    mem = load_memory(state.get("case_memory"))
    candidates = _cands(state)
    case = choose_final_case(candidates)
    if case is None:
        return {"response_type": "answer", "answer": "未找到匹配病例。",
                "confidence": 0.0, "phase": "ANSWERED",
                "turn_index": state.get("turn_index", 0) + 1,
                "case_memory": dump_memory(mem)}

    mem.resolved_case_id = case.case_id
    labels: list[str] = []
    for fid in mem.confirmed_features:
        lbl = mem.probe_labels.get(fid) or mem.probe_questions.get(fid) or fid
        if lbl and lbl not in labels:
            labels.append(lbl)
    if not labels:
        labels = case.matched_terms[:5] if case.matched_terms else [case.title]
    kf = "、".join(labels[:5]) or "暂无明确确认特征"
    answer = f"更符合本地病例：{case.title}\n关键依据：{kf}\n处理建议：{case.treat}"
    return {
        "response_type": "answer", "answer": answer, "treatment": case.treat,
        "matched_case": case.model_dump(), "resolved_case_id": case.case_id,
        "sources": [case.title], "confidence": confidence_from_candidates(candidates),
        "phase": "ANSWERED", "turn_index": state.get("turn_index", 0) + 1,
        "case_memory": dump_memory(mem),
    }
