"""Probe 评分与格式化工具。"""
from __future__ import annotations

import math
from hashlib import sha1
from typing import Sequence

from medical_assistant.text.split import norm_text, readable


def split_quality(pos: int, neg: int, unknown: int, total: int) -> float:
    """信息增益评分：一个 probe 把 total 切成 pos/neg/unknown 的质量。"""
    if total <= 0 or pos <= 0 or neg <= 0:
        return 0.0
    known = pos + neg
    coverage = known / total
    balance = 1.0 - abs(pos - neg) / max(1, known)
    unknown_penalty = 1.0 - 0.5 * (unknown / total)
    p = pos / known
    entropy = -(p * math.log2(p) + (1 - p) * math.log2(1 - p)) if 0 < p < 1 else 0.0
    return max(0.0, entropy * coverage * balance * unknown_penalty)


def make_probe_id(prefix: str, text: str) -> str:
    digest = sha1(norm_text(text).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def make_probe_label(text: str, max_len: int = 36) -> str:
    text = readable(text)
    return text if len(text) <= max_len else text[:max_len].rstrip("，、；;。")


def make_question_seed(
    prototype_text: str,
    evidence_texts: Sequence[str] | None = None,
) -> str:
    """生成追问文本。一个问题只问一个维度。"""
    evidence = [readable(x) for x in (evidence_texts or []) if readable(x)]
    candidates = sorted(evidence, key=len) if evidence else []
    proto = readable(prototype_text)
    if proto:
        candidates.append(proto)
    if not candidates:
        return "请补充更多症状描述。"
    body = next((c for c in candidates if len(c) >= 4), candidates[0])
    if len(body) > 60:
        body = body[:60].rstrip("，、；;。") + "……"
    return f"你的情况有没有这样：{body}？"


def best_threshold_split(
    sims_by_case: dict[str, float], *,
    margin: float = 0.02, min_child_size: int = 1,
) -> tuple[list[str], list[str], list[str], float, float]:
    """在相似度分布中找最优切分阈值。"""
    total = len(sims_by_case)
    if total <= 1:
        return list(sims_by_case), [], [], 0.0, 1.0
    values = sorted(set(round(v, 4) for v in sims_by_case.values()), reverse=True)
    if len(values) < 2:
        return list(sims_by_case), [], [], 0.0, values[0] if values else 0.0
    thresholds = [(values[i] + values[i + 1]) / 2 for i in range(len(values) - 1)]

    best = None
    best_score = -1.0
    for th in thresholds:
        pos = [cid for cid, s in sims_by_case.items() if s >= th + margin]
        unk = [cid for cid, s in sims_by_case.items() if th - margin <= s < th + margin]
        neg = [cid for cid, s in sims_by_case.items() if s < th - margin]
        if len(pos) < min_child_size or len(neg) < min_child_size:
            continue
        score = split_quality(len(pos), len(neg), len(unk), total)
        if score > best_score:
            best_score = score
            best = (pos, neg, unk, score, th)
    return best if best else ([], list(sims_by_case), [], 0.0, 1.0)
