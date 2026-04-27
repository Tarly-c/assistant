"""LLM Embedding 封装。带文本清洗和错误恢复。"""
from __future__ import annotations
import math, re
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


def _clean_for_embed(text: str) -> str:
    """清洗文本，防止 embedding 模型产出 NaN。"""
    text = (text or "").strip()
    # 去掉不可见字符和零宽字符
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
    text = re.sub(r"[\u200b-\u200f\u2028-\u202f\u2060\ufeff]", "", text)
    # 如果清洗后为空或太短，用占位文本（embedding 模型需要有内容）
    if not text or len(text.strip()) < 2:
        text = "空"
    return text


def _has_nan(vec: list[float]) -> bool:
    """检查向量是否包含 NaN 或 Inf。"""
    return any(math.isnan(v) or math.isinf(v) for v in vec)


def _zero_vec(dim: int = 768) -> list[float]:
    """返回零向量作为 fallback。"""
    return [0.0] * dim


def embed_one(text: str) -> list[float]:
    """单条 → 向量。"""
    text = _clean_for_embed(text)
    try:
        vec = _embedder().embed_query(text)
        return _zero_vec(len(vec)) if _has_nan(vec) else vec
    except Exception as e:
        print(f"[embed_one failed] {e}, text='{text[:50]}'")
        return _zero_vec()


def embed_batch(texts: Sequence[str], *, show_progress: bool = True) -> list[list[float]]:
    """批量 → 向量。带错误恢复：单条失败不影响整批。"""
    emb = _embedder()
    bs = get_settings().embedding_batch
    out: list[list[float]] = []
    dim = 768  # 会在第一批成功后更新

    for i in range(0, len(texts), bs):
        batch = [_clean_for_embed(t) for t in texts[i: i + bs]]
        try:
            vecs = emb.embed_documents(batch)
            if vecs:
                dim = len(vecs[0])
            # 检查每个向量
            cleaned = []
            for j, v in enumerate(vecs):
                if _has_nan(v):
                    print(f"  [WARN] NaN at index {i+j}, text='{batch[j][:40]}' → zero vec")
                    cleaned.append(_zero_vec(dim))
                else:
                    cleaned.append(v)
            out.extend(cleaned)
        except Exception as e:
            # 整批失败 → 逐条重试
            print(f"  [WARN] Batch {i}-{i+len(batch)} failed: {e}")
            print(f"         Retrying one-by-one...")
            for j, t in enumerate(batch):
                try:
                    v = emb.embed_query(t)
                    if _has_nan(v):
                        print(f"    [WARN] NaN at {i+j}, text='{t[:40]}' → zero vec")
                        out.append(_zero_vec(dim))
                    else:
                        out.append(v)
                        dim = len(v)
                except Exception as e2:
                    print(f"    [WARN] Single embed failed at {i+j}: {e2} → zero vec")
                    out.append(_zero_vec(dim))

        if show_progress and i > 0 and i % (bs * 10) == 0:
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
    if not vecs:
        return []
    dim = len(vecs[0])
    avg = [sum(v[d] for v in vecs) / len(vecs) for d in range(dim)]
    length = math.sqrt(sum(x * x for x in avg)) or 1.0
    return [x / length for x in avg]


def avg_best_match(query_vecs: list[list[float]],
                   target_vecs: list[list[float]]) -> float:
    if not query_vecs or not target_vecs:
        return 0.0
    scores = []
    for qv in query_vecs:
        best = max(cosine(qv, tv) for tv in target_vecs)
        scores.append(max(0.0, best))
    return sum(scores) / len(scores)
