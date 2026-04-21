from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from medical_assistant.config import get_settings
from medical_assistant.schemas.retrieval import LocalSearchResult, RetrievalHit


@lru_cache(maxsize=1)
def get_embeddings():
    settings = get_settings()
    return OllamaEmbeddings(model=settings.embedding_model)


@lru_cache(maxsize=1)
def get_vectorstore():
    settings = get_settings()
    settings.chroma_path.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=settings.collection_name,
        persist_directory=str(settings.chroma_path),
        embedding_function=get_embeddings(),
    )


def _lexical_score(text: str, normalized_terms: list[str]) -> float:
    if not normalized_terms:
        return 0.0

    haystack = text.lower()
    hits = sum(1 for term in normalized_terms if term and term.lower() in haystack)
    return hits / max(1, len(normalized_terms))


def search_local(
    question: str,
    local_query: str,
    normalized_terms: list[str] | None = None,
) -> dict:
    normalized_terms = normalized_terms or []
    settings = get_settings()
    vectorstore = get_vectorstore()

    try:
        docs_and_scores = vectorstore.similarity_search_with_relevance_scores(
            local_query or question,
            k=settings.local_top_k,
        )
    except Exception:
        docs_and_scores = [(doc, 0.0) for doc in vectorstore.similarity_search(local_query or question, k=settings.local_top_k)]

    hits: list[RetrievalHit] = []
    best_score = 0.0

    for doc, relevance_score in docs_and_scores:
        metadata = doc.metadata or {}
        source = str(metadata.get("source") or "")
        title = str(metadata.get("title") or Path(source).stem or "local document")
        chunk_id = metadata.get("chunk_id")
        snippet = (doc.page_content or "").strip()[:700]

        lexical = _lexical_score(f"{title}\n{snippet}", normalized_terms)
        score = round(0.7 * float(relevance_score) + 0.3 * lexical, 4)
        best_score = max(best_score, score)

        hits.append(
            RetrievalHit(
                source=source,
                title=title,
                chunk_id=int(chunk_id) if chunk_id is not None else None,
                snippet=snippet,
                score=score,
                metadata=metadata,
            )
        )

    hits.sort(key=lambda x: x.score, reverse=True)
    result = LocalSearchResult(
        enough=best_score >= settings.local_min_score,
        score=round(best_score, 4),
        reason="top local score compared with configured threshold",
        hits=hits[: settings.local_top_k],
    )
    return result.model_dump(mode="json")
