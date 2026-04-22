from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from medical_assistant.config import get_settings
from medical_assistant.schemas.input import SearchQuery
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


def _query_texts(
    question: str,
    queries: list[SearchQuery] | None = None,
    local_query: str | None = None,
) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()

    for item in queries or []:
        text = (item.text or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)

    fallback = (local_query or question or "").strip()
    if fallback:
        key = fallback.lower()
        if key not in seen:
            result.append(fallback)

    return result or [question]


def search_local(
    question: str,
    queries: list[SearchQuery] | None = None,
    normalized_terms: list[str] | None = None,
    local_query: str | None = None,
) -> dict:
    normalized_terms = normalized_terms or []
    settings = get_settings()
    vectorstore = get_vectorstore()
    query_texts = _query_texts(question=question, queries=queries, local_query=local_query)

    aggregated: dict[tuple[str, int | None, str], RetrievalHit] = {}
    best_score = 0.0

    for query_text in query_texts:
        try:
            docs_and_scores = vectorstore.similarity_search_with_relevance_scores(
                query_text,
                k=settings.local_top_k,
            )
        except Exception:
            docs_and_scores = [
                (doc, 0.0)
                for doc in vectorstore.similarity_search(query_text, k=settings.local_top_k)
            ]

        for doc, relevance_score in docs_and_scores:
            metadata = doc.metadata or {}
            source = str(metadata.get("source") or "")
            title = str(metadata.get("title") or Path(source).stem or "local document")
            chunk_id = metadata.get("chunk_id")
            chunk_id_int = int(chunk_id) if chunk_id is not None else None
            snippet = (doc.page_content or "").strip()[:700]
            lexical = _lexical_score(f"{title}\n{snippet}", normalized_terms)
            score = round(0.75 * float(relevance_score) + 0.25 * lexical, 4)
            best_score = max(best_score, score)

            key = (source, chunk_id_int, title)
            previous = aggregated.get(key)
            if previous is None or score > previous.score:
                aggregated[key] = RetrievalHit(
                    source=source,
                    title=title,
                    chunk_id=chunk_id_int,
                    snippet=snippet,
                    score=score,
                    metadata=metadata,
                )

    hits = sorted(aggregated.values(), key=lambda x: x.score, reverse=True)
    result = LocalSearchResult(
        enough=best_score >= settings.local_min_score,
        score=round(best_score, 4),
        reason="top local score compared with configured threshold",
        hits=hits[: settings.local_top_k],
    )
    return result.model_dump(mode="json")
