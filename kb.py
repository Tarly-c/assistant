from __future__ import annotations

import hashlib
import os
import re
import shutil
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

DEFAULT_DB_DIR = os.getenv("CHROMA_DIR", "chroma_db")
DEFAULT_COLLECTION = os.getenv("CHROMA_COLLECTION", "medical_kb")
DEFAULT_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]+|[a-zA-Z0-9]+")


def _embeddings(model: str | None = None) -> OllamaEmbeddings:
    return OllamaEmbeddings(model=model or DEFAULT_EMBED_MODEL)


def _vectorstore(
    persist_directory: str | None = None,
    collection_name: str | None = None,
    embedding_model: str | None = None,
) -> Chroma:
    return Chroma(
        collection_name=collection_name or DEFAULT_COLLECTION,
        persist_directory=persist_directory or DEFAULT_DB_DIR,
        embedding_function=_embeddings(embedding_model),
    )


def _read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages: List[str] = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages).strip()


def load_source_documents(resources_dir: str | Path = "resources") -> List[Document]:
    resources_path = Path(resources_dir)
    docs: List[Document] = []

    if not resources_path.exists():
        return docs

    for path in resources_path.rglob("*"):
        if not path.is_file():
            continue

        ext = path.suffix.lower()
        try:
            if ext in {".txt", ".md"}:
                text = path.read_text(encoding="utf-8", errors="ignore").strip()
            elif ext == ".pdf":
                text = _read_pdf(path)
            else:
                continue
        except Exception as exc:
            print(f"[kb] 跳过 {path}: {exc}")
            continue

        if not text:
            continue

        docs.append(
            Document(
                page_content=text,
                metadata={"source": str(path)},
            )
        )

    return docs


def _add_documents_batched(store, docs, ids, preferred_batch_size: int = 1000) -> None:
    """
    分批写入 Chroma，避免超过 max batch size。
    preferred_batch_size 用 1000 就很稳。
    """
    try:
        # langchain_chroma 内部持有 chroma client
        max_batch_size = store._client.get_max_batch_size()
        batch_size = max(1, min(preferred_batch_size, max_batch_size))
    except Exception:
        # 取不到就用一个保守值
        batch_size = preferred_batch_size

    total = len(docs)
    print(f"[kb] total chunks={total}, batch_size={batch_size}")

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        store.add_documents(docs[start:end], ids=ids[start:end])
        print(f"[kb] indexed {end}/{total}")


def build_index(
    resources_dir: str | Path = "resources",
    persist_directory: str = DEFAULT_DB_DIR,
    collection_name: str = DEFAULT_COLLECTION,
    embedding_model: str = DEFAULT_EMBED_MODEL,
    reset: bool = False,
) -> int:
    if reset and Path(persist_directory).exists():
        shutil.rmtree(persist_directory)

    docs = load_source_documents(resources_dir)
    if not docs:
        raise RuntimeError("resources/ 为空，或没有可读取的 .txt/.md/.pdf 文件。")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(docs)

    for i, doc in enumerate(chunks):
        doc.metadata["chunk_id"] = i

    store = _vectorstore(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_model=embedding_model,
    )

    ids = []
    for i, doc in enumerate(chunks):
        raw = f"{doc.metadata.get('source', '')}::{doc.metadata.get('chunk_id', i)}::{doc.page_content[:120]}"
        ids.append(hashlib.md5(raw.encode("utf-8")).hexdigest())

    _add_documents_batched(store, chunks, ids)
    return len(chunks)


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall((text or "").lower()))


def _lexical_score(query: str, text: str) -> float:
    q = _tokenize(query)
    t = _tokenize(text)
    if not q or not t:
        return 0.0

    overlap = len(q & t) / max(len(q), 1)
    phrase_bonus = 0.25 if query and query.lower() in text.lower() else 0.0
    return round(overlap + phrase_bonus, 4)


def search_local(
    question: str,
    local_query: str,
    persist_directory: str = DEFAULT_DB_DIR,
    collection_name: str = DEFAULT_COLLECTION,
    embedding_model: str = DEFAULT_EMBED_MODEL,
    k: int = 6,
    min_score: float = 0.18,
) -> dict:
    db_path = Path(persist_directory)
    if not db_path.exists():
        return {
            "enough": False,
            "score": 0.0,
            "hits": [],
            "reason": "NO_LOCAL_INDEX",
        }

    store = _vectorstore(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_model=embedding_model,
    )

    try:
        docs = store.similarity_search(local_query or question, k=k)
    except Exception as exc:
        return {
            "enough": False,
            "score": 0.0,
            "hits": [],
            "reason": f"LOCAL_SEARCH_ERROR: {exc}",
        }

    hits = []
    for doc in docs:
        content = (doc.page_content or "").strip()
        score = max(
            _lexical_score(local_query, content),
        )
        hits.append(
            {
                "source": doc.metadata.get("source", ""),
                "chunk_id": doc.metadata.get("chunk_id"),
                "score": score,
                "snippet": content[:700],
            }
        )

    hits.sort(key=lambda x: x["score"], reverse=True)
    top_score = hits[0]["score"] if hits else 0.0
    enough = bool(hits) and top_score >= min_score

    return {
        "enough": enough,
        "score": top_score,
        "hits": hits[:3],
        "reason": None if enough else "LOW_LOCAL_CONFIDENCE",
    }
