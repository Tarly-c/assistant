"""Probe 评分 + LLM 改写。"""
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


def rephrase(description: str, evidence: list[str] | None = None) -> str:
    """★ LLM 改写追问。失败回退模板。"""
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


def best_threshold(
    values: dict[str, float], *, margin: float = 0.02, min_child: int = 1,
) -> tuple[list[str], list[str], list[str], float, float]:
    """在数值分布中找最优二分阈值。"""
    total = len(values)
    if total <= 1:
        return list(values), [], [], 0.0, 1.0
    vals = sorted(set(round(v, 4) for v in values.values()), reverse=True)
    if len(vals) < 2:
        return list(values), [], [], 0.0, vals[0] if vals else 0.0
    best, best_s = None, -1.0
    for th in [(vals[i] + vals[i+1]) / 2 for i in range(len(vals) - 1)]:
        pos = [c for c, s in values.items() if s >= th + margin]
        unk = [c for c, s in values.items() if th - margin <= s < th + margin]
        neg = [c for c, s in values.items() if s < th - margin]
        if len(pos) < min_child or len(neg) < min_child:
            continue
        sq = split_quality(len(pos), len(neg), len(unk), total)
        if sq > best_s:
            best_s = sq
            best = (pos, neg, unk, sq, th)
    return best if best else ([], list(values), [], 0.0, 1.0)
