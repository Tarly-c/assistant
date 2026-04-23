"""本地 Chroma 向量检索。"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from medical_assistant.config import get_settings


@lru_cache(maxsize=1)
def _get_embeddings():
    settings = get_settings()
    return OllamaEmbeddings(model=settings.embedding_model)


@lru_cache(maxsize=1)
def _get_vectorstore():
    settings = get_settings()
    settings.chroma_path.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=settings.collection_name,
        persist_directory=str(settings.chroma_path),
        embedding_function=_get_embeddings(),
    )


def _lexical_boost(text: str, terms: list[str]) -> float:
    """关键词命中率，0~1。"""
    if not terms:
        return 0.0
    haystack = text.lower()
    hits = sum(1 for t in terms if t and t.lower() in haystack)
    return hits / max(1, len(terms))


def search_local(
    question: str,
    normalized_terms: list[str] | None = None,
) -> dict:
    """
    返回 dict:
      enough: bool
      score: float
      reason: str
      hits: list[dict]  — 每个 dict 含 source, title, snippet, score, chunk_id
    """
    terms = normalized_terms or []
    settings = get_settings()
    vs = _get_vectorstore()

    try:
        docs_and_scores = vs.similarity_search_with_relevance_scores(
            question, k=settings.local_top_k,
        )
    except Exception:
        docs_and_scores = [
            (doc, 0.0)
            for doc in vs.similarity_search(question, k=settings.local_top_k)
        ]

    hits: list[dict] = []
    best = 0.0

    for doc, rel_score in docs_and_scores:
        meta = doc.metadata or {}
        title = str(meta.get("title") or Path(str(meta.get("source", ""))).stem or "unknown")
        snippet = (doc.page_content or "").strip()[:700]
        lexical = _lexical_boost(f"{title}\n{snippet}", terms)
        score = round(0.75 * float(rel_score) + 0.25 * lexical, 4)
        best = max(best, score)

        hits.append({
            "source": str(meta.get("source", "")),
            "title": title,
            "snippet": snippet,
            "score": score,
            "chunk_id": str(meta.get("chunk_id", "")),
        })

    hits.sort(key=lambda x: x["score"], reverse=True)
    hits = hits[: settings.local_top_k]

    return {
        "enough": best >= settings.local_min_score,
        "score": round(best, 4),
        "reason": f"best={best:.4f} vs threshold={settings.local_min_score}",
        "hits": hits,
    }
