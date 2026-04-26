"""TF-IDF 字符 n-gram 向量化 + 余弦相似度。不含业务逻辑。"""
from __future__ import annotations

import math
import re
from typing import Sequence

from medical_assistant.text.split import norm_text


def char_ngrams(text: str, ns: tuple[int, ...] = (2, 3)) -> list[str]:
    compact = norm_text(text)
    grams: list[str] = []
    for n in ns:
        if len(compact) >= n:
            grams.extend(compact[i: i + n] for i in range(len(compact) - n + 1))
    return grams


def build_idf(texts: Sequence[str]) -> dict[str, float]:
    doc_count: dict[str, int] = {}
    for text in texts:
        for gram in set(char_ngrams(text)):
            doc_count[gram] = doc_count.get(gram, 0) + 1
    n = max(1, len(texts))
    return {g: math.log((1 + n) / (1 + df)) + 1.0 for g, df in doc_count.items()}


def vectorize(text: str, idf: dict[str, float] | None = None) -> dict[str, float]:
    idf = idf or {}
    counts: dict[str, float] = {}
    for g in char_ngrams(text):
        counts[g] = counts.get(g, 0.0) + 1.0
    weighted = {g: c * idf.get(g, 1.0) for g, c in counts.items()}
    length = math.sqrt(sum(v * v for v in weighted.values())) or 1.0
    return {g: v / length for g, v in weighted.items()}


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(k, 0.0) for k, v in a.items())
