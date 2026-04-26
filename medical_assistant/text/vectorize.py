"""TF-IDF 字符 n-gram 向量化 + 余弦相似度。"""
from __future__ import annotations

import math
from typing import Sequence
from medical_assistant.text.split import norm


def char_ngrams(text: str, ns: tuple[int, ...] = (2, 3)) -> list[str]:
    c = norm(text)
    grams = []
    for n in ns:
        if len(c) >= n:
            grams.extend(c[i: i + n] for i in range(len(c) - n + 1))
    return grams


def build_idf(texts: Sequence[str]) -> dict[str, float]:
    df: dict[str, int] = {}
    for t in texts:
        for g in set(char_ngrams(t)):
            df[g] = df.get(g, 0) + 1
    n = max(1, len(texts))
    return {g: math.log((1 + n) / (1 + c)) + 1.0 for g, c in df.items()}


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
