"""Memory 读写。"""
from __future__ import annotations
from typing import Any
from medical_assistant.schemas import Memory, Probe
from medical_assistant.session.parser import parse_answer


def load(raw: dict | Memory | None) -> Memory:
    if isinstance(raw, Memory):
        return raw
    return Memory(**(raw or {})) if isinstance(raw, dict) else Memory()


def dump(mem: Memory) -> dict[str, Any]:
    return mem.model_dump()


def _merge(a: list[str], b: list[str]) -> list[str]:
    out = list(a or [])
    for x in b:
        if x and x not in out:
            out.append(x)
    return out


def process_input(
    mem: Memory, *,
    user_input: str,
    query_cn: str, query_en: str,
    query_sentence_vec: list[float],
    query_keyword_vecs: list[list[float]],
    intent: str, turn: int,
) -> Memory:
    """根据用户输入更新记忆。"""
    is_answering = bool(mem.last_probe_id)

    if not mem.original_input:
        # 首轮
        mem.original_input = user_input
        mem.query_cn = query_cn
        mem.query_en = query_en
        mem.query_sentence_vec = query_sentence_vec
        mem.query_keyword_vecs = query_keyword_vecs
        mem.intent = intent
    elif not is_answering:
        # 新问题
        mem.query_cn = query_cn
        mem.query_en = query_en
        mem.query_sentence_vec = query_sentence_vec
        mem.query_keyword_vecs = query_keyword_vecs
        mem.intent = intent
    # else: 回答追问 → 保持原始查询向量不变

    mem.turn = turn

    if is_answering:
        pid = mem.last_probe_id
        parsed = parse_answer(
            probe_text=mem.last_probe_text,
            user_input=user_input,
            probe_label=mem.labels.get(pid, pid),
        )
        if parsed.signal == "yes":
            mem.confirmed[pid] = {"evidence": parsed.evidence}
            mem.denied.pop(pid, None); mem.uncertain.pop(pid, None)
            if mem.last_yes_child:
                mem.tree_node = mem.last_yes_child
        elif parsed.signal == "no":
            mem.denied[pid] = {"evidence": parsed.evidence}
            mem.confirmed.pop(pid, None); mem.uncertain.pop(pid, None)
            if mem.last_no_child:
                mem.tree_node = mem.last_no_child
        elif parsed.signal == "uncertain":
            mem.uncertain[pid] = {"evidence": parsed.evidence}
            if mem.last_tree_node:
                mem.tree_node = mem.last_tree_node

        # 用户回答可能夹带新概念 → 追加关键词向量
        if parsed.new_observations:
            from medical_assistant.text.embed import embed_batch
            new_vecs = embed_batch(parsed.new_observations)
            mem.query_keyword_vecs = mem.query_keyword_vecs + new_vecs

    return mem


def record_probe(mem: Memory, probe: Probe) -> Memory:
    mem.last_probe_id = probe.probe_id
    mem.last_probe_text = probe.text
    mem.last_probe_label = probe.label
    mem.last_tree_node = probe.tree_node
    mem.last_yes_child = probe.yes_child
    mem.last_no_child = probe.no_child
    if probe.tree_node:
        mem.tree_node = probe.tree_node
    mem.asked_probes = _merge(mem.asked_probes, [probe.probe_id])
    mem.splits[probe.probe_id] = {
        "positive": probe.positive_ids,
        "negative": probe.negative_ids,
        "unknown": probe.unknown_ids,
    }
    mem.labels[probe.probe_id] = probe.label
    return mem
