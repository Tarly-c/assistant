"""5 个 workflow 节点。★ 在线概念抽取 + N-自适应评分。"""
from __future__ import annotations

from medical_assistant.config import get_settings
from medical_assistant.graph.state import S
from medical_assistant.llm import call_structured
from medical_assistant.prompts import NORMALIZE
from medical_assistant.schemas import Memory, NormalizedInput, ScoredCase
from medical_assistant.text.embed import embed_one, embed_batch
from medical_assistant.cases.store import all_ids
from medical_assistant.cases.ranker import confidence, filter_by_probes, rank_cases
from medical_assistant.session.memory import dump, load, process_input, record_probe
from medical_assistant.session.planner import pick_probe, should_stop
from medical_assistant.tree.navigator import load_tree


# ── 节点 1：解析输入 + 更新记忆 ──

def parse_input(state: S) -> dict:
    """★ 首轮/新问题：LLM 双语抽取 + 概念抽取 + embed。
    回答追问：沿用原始向量，可能追加新概念。
    """
    user_input = state.get("user_input", "")
    mem = load(state.get("memory"))
    is_answering = bool(mem.last_probe_id)

    if is_answering:
        # 回答追问 → 不替换查询向量
        query_cn = mem.query_cn or mem.original_input
        query_en = mem.query_en
        sent_vec = mem.query_sentence_vec
        kw_vecs = list(mem.query_keyword_vecs)
        intent = mem.intent
    else:
        # 首轮/新问题 → LLM 抽取 + embed
        llm_out = call_structured(NormalizedInput, [
            {"role": "system", "content": NORMALIZE},
            {"role": "user", "content": f"当前问题: {user_input}"},
        ])
        query_cn = llm_out.query_cn or user_input
        query_en = llm_out.query_en or ""
        intent = llm_out.intent or "general"

        # ★ embed 句子向量（用完整输入）
        sent_vec = embed_one(query_cn)

        # ★ embed 关键词向量（对每个 concept.term 做 embed）
        concept_terms = [c.term for c in llm_out.concepts if c.term]
        kw_vecs = embed_batch(concept_terms) if concept_terms else []

    mem = process_input(
        mem, user_input=user_input,
        query_cn=query_cn, query_en=query_en,
        query_sentence_vec=sent_vec, query_keyword_vecs=kw_vecs,
        intent=intent, turn=state.get("turn", 0),
    )
    return {"memory": dump(mem)}


# ── 节点 2：缩窄候选 ──

def narrow(state: S) -> dict:
    mem = load(state.get("memory"))
    tree = load_tree()
    tree_ids = []
    if mem.tree_node and tree:
        node = (tree.get("nodes") or {}).get(mem.tree_node)
        if isinstance(node, dict):
            tree_ids = list(node.get("case_ids", []))
    base = tree_ids or mem.candidate_ids or all_ids()
    filtered = filter_by_probes(
        base, confirmed=mem.confirmed.keys(),
        denied=mem.denied.keys(), splits=mem.splits,
    ) or base
    mem.candidate_ids = filtered
    return {"memory": dump(mem)}


# ── 节点 3：★ N-自适应三路评分 ──

def score(state: S) -> dict:
    cfg = get_settings()
    mem = load(state.get("memory"))

    ranked = rank_cases(
        query_sentence_vec=mem.query_sentence_vec,
        query_keyword_vecs=mem.query_keyword_vecs,
        candidate_ids=mem.candidate_ids,
        confirmed=mem.confirmed.keys(),
        denied=mem.denied.keys(),
        splits=mem.splits,
    )

    mem.candidate_ids = [c.case_id for c in ranked]
    bs = ranked[0].score if ranked else 0.0
    conf = confidence(ranked)
    top = [{"case_id": c.case_id, "title": c.title, "score": c.score,
            "sentence_sim": c.sentence_sim, "keyword_sim": c.keyword_sim,
            "probe_score": c.probe_score, "hit_probes": c.hit_probes}
           for c in ranked[:cfg.display_top_k]]

    return {
        "candidates": [c.model_dump() for c in ranked],
        "candidate_count": len(ranked),
        "best_score": bs, "confidence": conf,
        "top_candidates": top, "memory": dump(mem),
    }


# ── 路由 ──

def _to_scored(state: S) -> list[ScoredCase]:
    return [ScoredCase(**x) for x in state.get("candidates", [])]


def route(state: S) -> str:
    mem = load(state.get("memory"))
    return "answer" if should_stop(_to_scored(state), mem) else "ask"


# ── 节点 4：追问 ──

def ask(state: S) -> dict:
    mem = load(state.get("memory"))
    probe = pick_probe(_to_scored(state), mem)
    if probe is None:
        return {"reply": "请补充更多信息。", "reply_type": "answer"}
    mem = record_probe(mem, probe)
    return {
        "reply": probe.text, "reply_type": "question",
        "probe": probe.model_dump(), "memory": dump(mem),
        "turn": state.get("turn", 0) + 1,
    }


# ── 节点 5：结论 ──

def answer(state: S) -> dict:
    mem = load(state.get("memory"))
    candidates = _to_scored(state)
    if not candidates:
        return {"reply": "未找到匹配病例。", "reply_type": "answer",
                "confidence": 0.0, "memory": dump(mem),
                "turn": state.get("turn", 0) + 1}

    case = candidates[0]
    mem.resolved_case_id = case.case_id
    reasons = [mem.labels.get(p, p) for p in mem.confirmed if mem.labels.get(p)][:5]
    if not reasons:
        reasons = [case.title]

    reply = (f"更符合本地病例：{case.title}\n"
             f"关键依据：{'、'.join(reasons)}\n"
             f"处理建议：{case.treat}")
    return {
        "reply": reply, "reply_type": "answer",
        "matched_case": case.model_dump(),
        "confidence": confidence(candidates),
        "memory": dump(mem),
        "turn": state.get("turn", 0) + 1,
    }
