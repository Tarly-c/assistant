"""LLM Embedding 封装。所有向量化都走这里。"""
from __future__ import annotations
import math
from functools import lru_cache
from typing import Sequence
from medical_assistant.config import get_settings

try:
    from langchain_ollama import OllamaEmbeddings
except Exception:
    OllamaEmbeddings = None


@lru_cache(maxsize=1)
def _embedder():
    if OllamaEmbeddings is None:
        raise RuntimeError("langchain_ollama not installed")
    return OllamaEmbeddings(model=get_settings().embedding_model)


def embed_one(text: str) -> list[float]:
    """单条 → 向量。在线用。"""
    return _embedder().embed_query(text)


def embed_batch(texts: Sequence[str]) -> list[list[float]]:
    """批量 → 向量。离线用。"""
    emb = _embedder()
    bs = get_settings().embedding_batch
    out = []
    for i in range(0, len(texts), bs):
        out.extend(emb.embed_documents(list(texts[i: i + bs])))
        if i > 0 and i % (bs * 10) == 0:
            print(f"    embedded {i}/{len(texts)}...")
    return out


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


def mean_vec(vecs: list[list[float]]) -> list[float]:
    """均值后 L2 归一化。"""
    if not vecs:
        return []
    dim = len(vecs[0])
    avg = [sum(v[d] for v in vecs) / len(vecs) for d in range(dim)]
    length = math.sqrt(sum(x * x for x in avg)) or 1.0
    return [x / length for x in avg]


def avg_best_match(query_vecs: list[list[float]],
                   target_vecs: list[list[float]]) -> float:
    """对 query 中每个向量，在 target 中找最佳匹配，返回平均值。"""
    if not query_vecs or not target_vecs:
        return 0.0
    scores = []
    for qv in query_vecs:
        best = max(cosine(qv, tv) for tv in target_vecs)
        scores.append(max(0.0, best))
    return sum(scores) / len(scores)
