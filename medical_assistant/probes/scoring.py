"""Probe 评分与格式化工具。"""
from __future__ import annotations

import math
from hashlib import sha1
from typing import Sequence

from medical_assistant.text.split import norm, clean


def split_quality(pos: int, neg: int, unk: int, total: int) -> float:
    """一个 probe 把 total 切成 pos/neg/unk 的信息增益评分。"""
    if total <= 0 or pos <= 0 or neg <= 0:
        return 0.0
    known = pos + neg
    p = pos / known
    entropy = -(p * math.log2(p) + (1 - p) * math.log2(1 - p)) if 0 < p < 1 else 0.0
    coverage = known / total
    balance = 1.0 - abs(pos - neg) / max(1, known)
    unk_penalty = 1.0 - 0.5 * (unk / total)
    return max(0.0, entropy * coverage * balance * unk_penalty)


def probe_id(prefix: str, text: str) -> str:
    return f"{prefix}_{sha1(norm(text).encode()).hexdigest()[:12]}"


def probe_label(text: str, max_len: int = 36) -> str:
    t = clean(text)
    return t if len(t) <= max_len else t[:max_len].rstrip("，、；;。")


def probe_question(prototype: str, evidence: Sequence[str] | None = None) -> str:
    """生成追问文本。一个问题只问一个维度，≤60 字。"""
    candidates = sorted([clean(x) for x in (evidence or []) if clean(x)], key=len)
    proto = clean(prototype)
    if proto:
        candidates.append(proto)
    if not candidates:
        return "请补充更多症状描述。"
    body = next((c for c in candidates if len(c) >= 4), candidates[0])
    if len(body) > 60:
        body = body[:60].rstrip("，、；;。") + "……"
    return f"你的情况有没有这样：{body}？"


def best_threshold(
    sims: dict[str, float], *, margin: float = 0.02, min_child: int = 1,
) -> tuple[list[str], list[str], list[str], float, float]:
    """在相似度分布中找最优二分阈值。"""
    total = len(sims)
    if total <= 1:
        return list(sims), [], [], 0.0, 1.0
    vals = sorted(set(round(v, 4) for v in sims.values()), reverse=True)
    if len(vals) < 2:
        return list(sims), [], [], 0.0, vals[0] if vals else 0.0

    best, best_s = None, -1.0
    for th in [(vals[i] + vals[i+1]) / 2 for i in range(len(vals) - 1)]:
        pos = [c for c, s in sims.items() if s >= th + margin]
        unk = [c for c, s in sims.items() if th - margin <= s < th + margin]
        neg = [c for c, s in sims.items() if s < th - margin]
        if len(pos) < min_child or len(neg) < min_child:
            continue
        sq = split_quality(len(pos), len(neg), len(unk), total)
        if sq > best_s:
            best_s = sq
            best = (pos, neg, unk, sq, th)
    return best if best else ([], list(sims), [], 0.0, 1.0)
