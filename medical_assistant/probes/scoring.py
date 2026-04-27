"""Probe 评分 + 语义切分 + LLM 改写。"""
from __future__ import annotations

import math
from medical_assistant.text.split import clean
from medical_assistant.llm import call_text
from medical_assistant.prompts import REPHRASE_PROBE


def split_quality(pos: int, neg: int, unk: int, total: int) -> float:
    """probe 切分信息增益。"""
    if total <= 0 or pos <= 0 or neg <= 0:
        return 0.0
    known = pos + neg
    p = pos / known
    entropy = -(p * math.log2(p) + (1 - p) * math.log2(1 - p)) if 0 < p < 1 else 0.0
    return max(0.0, entropy
               * (known / total)
               * (1.0 - abs(pos - neg) / max(1, known))
               * (1.0 - 0.5 * (unk / total)))


def semantic_split(
    values: dict[str, float],
    *,
    anchor: float = 0.50,
    search_range: float = 0.12,
    margin: float = 0.03,
    min_child: int = 1,
) -> tuple[list[str], list[str], list[str], float, float]:
    """在语义边界附近找切分点。

    只考虑 [anchor - search_range, anchor + search_range] 范围内的阈值。
    如果数据不跨越这个范围 → 返回空（这个维度没有区分力）。
    """
    total = len(values)
    if total <= 1:
        return [], list(values), [], 0.0, anchor

    lo = anchor - search_range
    hi = anchor + search_range

    # 预检：数据是否跨越语义边界
    has_above = any(v >= hi for v in values.values())
    has_below = any(v <= lo for v in values.values())
    if not has_above or not has_below:
        return [], list(values), [], 0.0, anchor

    # 只在 [lo, hi] 范围内的值之间搜索阈值
    candidates = sorted(set(
        round(v, 4) for v in values.values() if lo <= v <= hi
    ))

    if not candidates:
        thresholds = [anchor]
    else:
        thresholds = []
        for i in range(len(candidates) - 1):
            thresholds.append((candidates[i] + candidates[i + 1]) / 2)
        if anchor not in thresholds:
            thresholds.append(anchor)

    best, best_s = None, -1.0
    for th in thresholds:
        pos = [c for c, v in values.items() if v >= th + margin]
        unk = [c for c, v in values.items() if th - margin <= v < th + margin]
        neg = [c for c, v in values.items() if v < th - margin]
        if len(pos) < min_child or len(neg) < min_child:
            continue
        sq = split_quality(len(pos), len(neg), len(unk), total)
        if sq > best_s:
            best_s = sq
            best = (pos, neg, unk, sq, th)

    return best if best else ([], list(values), [], 0.0, anchor)


def rephrase(description: str, evidence: list[str] | None = None) -> str:
    """LLM 改写追问。失败回退模板。"""
    ev = "；".join(clean(x) for x in (evidence or []) if clean(x))[:200]
    result = call_text([{"role": "user",
        "content": REPHRASE_PROBE.format(description=description, evidence=ev or "无")}])
    result = clean(result)
    if result and len(result) >= 6 and "？" in result:
        return result
    body = clean(description)
    if len(body) > 60:
        body = body[:60] + "……"
    return f"你的情况有没有这样：{body}？"
