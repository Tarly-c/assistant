"""5 个 workflow 节点。每个节点接收 S、返回 partial S。"""
from __future__ import annotations

from medical_assistant.config import get_settings
from medical_assistant.graph.state import S
from medical_assistant.llm import call_structured
from medical_assistant.prompts import NORMALIZE
from medical_assistant.schemas import Memory, NormalizedInput, Probe, ScoredCase
from medical_assistant.text.split import extract_terms
from medical_assistant.cases.store import all_ids
from medical_assistant.cases.ranker import confidence, filter_by_probes, rank_cases
from medical_assistant.session.memory import dump, load, process_user_input, record_probe
from medical_assistant.session.planner import pick_probe, should_stop
from medical_assistant.tree.navigator import load_tree


# ── 节点 1：解析输入 + 更新记忆 ──

def parse_input(state: S) -> dict:
    """合并了 normalize + update_memory。

    首轮/新问题 → LLM 抽取搜索信息。
    回答追问 → 解析回答，沿用原始搜索查询。
    """
    user_input = state.get("user_input", "")
    mem = load(state.get("memory"))
    is_answering = bool(mem.last_probe_id)

    if is_answering:
        # 用户在回答追问 → 不替换搜索查询
        query = mem.search_query or mem.original_input or user_input
        terms = list(mem.search_terms)
        intent = mem.intent
        # 但可能夹带新信息（"不是，我是晚上自己疼"）
        for t in extract_terms(user_input):
            if len(t) >= 3 and t not in terms:
                terms.append(t)
    else:
        # 首轮 / 新问题 → LLM 结构化抽取
        llm_out = call_structured(NormalizedInput, [
            {"role": "system", "content": NORMALIZE},
            {"role": "user", "content": f"当前问题: {user_input}"},
        ])
        query = llm_out.query_en or user_input
        intent = llm_out.intent or "general"
        terms = list(llm_out.key_terms or [])
        for t in extract_terms(user_input):
            if t and t not in terms:
                terms.append(t)

    # 更新记忆
    mem = process_user_input(
        mem, user_input=user_input, search_query=query,
        search_terms=terms[:15], intent=intent, turn=state.get("turn", 0),
    )
    return {"memory": dump(mem)}


# ── 节点 2：缩窄候选集 ──

def narrow(state: S) -> dict:
    """根据树节点 + confirmed/denied 过滤候选 ID 列表。"""
    mem = load(state.get("memory"))

    # 优先用树节点的病例集
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


# ── 节点 3：评分排序 ──

def score(state: S) -> dict:
    """对缩窄后的候选集评分排序。"""
    cfg = get_settings()
    mem = load(state.get("memory"))

    # 始终用原始问题做搜索，不用"是的"/"不是"
    query = mem.search_query or mem.original_input
    terms = mem.search_terms

    ranked = rank_cases(
        query, terms, candidate_ids=mem.candidate_ids,
        confirmed=mem.confirmed.keys(), denied=mem.denied.keys(),
        splits=mem.splits,
    )

    mem.candidate_ids = [c.case_id for c in ranked]
    bs = ranked[0].score if ranked else 0.0
    conf = confidence(ranked)
    top = [{"case_id": c.case_id, "title": c.title, "score": c.score,
            "hit_probes": c.hit_probes, "hit_terms": c.hit_terms}
           for c in ranked[:cfg.display_top_k]]

    return {
        "candidates": [c.model_dump() for c in ranked],
        "candidate_count": len(ranked),
        "best_score": bs,
        "confidence": conf,
        "top_candidates": top,
        "memory": dump(mem),
    }


# ── 路由：继续问 or 给答案 ──

def _to_scored(state: S) -> list[ScoredCase]:
    return [ScoredCase(**x) for x in state.get("candidates", [])]


def route(state: S) -> str:
    """决定下一步：'ask' 继续追问 / 'answer' 给出结论。"""
    mem = load(state.get("memory"))
    return "answer" if should_stop(_to_scored(state), mem) else "ask"


# ── 节点 4：选择追问 ──

def ask(state: S) -> dict:
    """选择最佳追问，记录到记忆中。"""
    mem = load(state.get("memory"))
    candidates = _to_scored(state)
    probe = pick_probe(candidates, mem)

    if probe is None:
        # 无问题可问 → 应该走 answer，但作为兜底
        return {"reply": "请补充更多信息。", "reply_type": "answer"}

    mem = record_probe(mem, probe)
    return {
        "reply": probe.text,
        "reply_type": "question",
        "probe": probe.model_dump(),
        "memory": dump(mem),
        "turn": state.get("turn", 0) + 1,
    }


# ── 节点 5：给出结论 ──

def answer(state: S) -> dict:
    """选出最佳病例，格式化最终答案。"""
    mem = load(state.get("memory"))
    candidates = _to_scored(state)

    if not candidates:
        return {"reply": "未找到匹配病例。", "reply_type": "answer",
                "confidence": 0.0, "memory": dump(mem),
                "turn": state.get("turn", 0) + 1}

    case = candidates[0]
    mem.resolved_case_id = case.case_id

    # 收集确认过的 probe 标签作为依据
    reasons = []
    for pid in mem.confirmed:
        lbl = mem.labels.get(pid, pid)
        if lbl and lbl not in reasons:
            reasons.append(lbl)
    if not reasons:
        reasons = case.hit_terms[:5] if case.hit_terms else [case.title]

    reply = (f"更符合本地病例：{case.title}\n"
             f"关键依据：{'、'.join(reasons[:5]) or '暂无'}\n"
             f"处理建议：{case.treat}")

    return {
        "reply": reply,
        "reply_type": "answer",
        "matched_case": case.model_dump(),
        "confidence": confidence(candidates),
        "memory": dump(mem),
        "turn": state.get("turn", 0) + 1,
    }
