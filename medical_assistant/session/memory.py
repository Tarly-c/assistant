"""Memory 读写与跨轮更新。"""
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


def _merge(existing: list[str], new: list[str]) -> list[str]:
    out = list(existing or [])
    for item in new:
        if item and item not in out:
            out.append(item)
    return out


def process_user_input(
    mem: Memory, *,
    user_input: str,
    search_query: str,
    search_terms: list[str],
    intent: str,
    turn: int,
) -> Memory:
    """根据用户本轮输入更新记忆。

    如果用户在回答上一轮追问 → 解析回答，更新确认/否认状态。
    如果是首轮/新问题 → 更新搜索信息。
    """
    is_answering = bool(mem.last_probe_id)

    # 首轮：记录原始信息
    if not mem.original_input:
        mem.original_input = user_input
        mem.search_query = search_query
        mem.intent = intent
    elif not is_answering:
        # 用户换了话题
        mem.search_query = search_query
        mem.intent = intent

    mem.search_terms = _merge(mem.search_terms, search_terms)
    mem.turn = turn

    # 解析上一轮追问的回答
    if is_answering:
        parsed = parse_answer(
            probe_text=mem.last_probe_text,
            user_input=user_input,
            probe_label=mem.labels.get(mem.last_probe_id, mem.last_probe_id),
        )
        pid = mem.last_probe_id
        sig = parsed.signal

        if sig == "yes":
            mem.confirmed[pid] = {"evidence": parsed.evidence}
            mem.denied.pop(pid, None)
            mem.uncertain.pop(pid, None)
            if mem.last_yes_child:
                mem.tree_node = mem.last_yes_child
        elif sig == "no":
            mem.denied[pid] = {"evidence": parsed.evidence}
            mem.confirmed.pop(pid, None)
            mem.uncertain.pop(pid, None)
            if mem.last_no_child:
                mem.tree_node = mem.last_no_child
        elif sig == "uncertain":
            mem.uncertain[pid] = {"evidence": parsed.evidence}
            if mem.last_tree_node:
                mem.tree_node = mem.last_tree_node

        # 用户回答里夹带的新症状
        mem.search_terms = _merge(mem.search_terms, parsed.new_observations)

    return mem


def record_probe(mem: Memory, probe: Probe) -> Memory:
    """记录本轮选中的追问，供下一轮解析回答。"""
    mem.last_probe_id = probe.probe_id
    mem.last_probe_text = probe.text
    mem.last_probe_label = probe.label
    mem.last_tree_node = probe.tree_node
    mem.last_yes_child = probe.yes_child
    mem.last_no_child = probe.no_child

    if probe.tree_node:
        mem.tree_node = probe.tree_node

    mem.asked_probes = _merge(mem.asked_probes, [probe.probe_id])

    # 保存切分数据
    mem.splits[probe.probe_id] = {
        "positive": probe.positive_ids,
        "negative": probe.negative_ids,
        "unknown": probe.unknown_ids,
    }
    mem.labels[probe.probe_id] = probe.label
    return mem


def clear_last_probe(mem: Memory) -> Memory:
    """清除上一轮追问（进入非追问状态）。"""
    mem.last_probe_id = ""
    mem.last_probe_text = ""
    return mem
