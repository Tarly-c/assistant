from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from medical_assistant.config import get_settings
from medical_assistant.services.retrieval.local import get_embeddings


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf_file(path: Path) -> str:
    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts)


def load_documents(resources_dir: Path) -> list[Document]:
    docs: list[Document] = []

    for path in resources_dir.rglob("*"):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()
        if suffix not in {".md", ".txt", ".pdf"}:
            continue

        if suffix == ".pdf":
            text = read_pdf_file(path)
        else:
            text = read_text_file(path)

        if not text.strip():
            continue

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(path.as_posix()),
                    "title": path.stem.replace("_", " "),
                },
            )
        )

    return docs


def main() -> None:
    settings = get_settings()
    settings.chroma_path.mkdir(parents=True, exist_ok=True)

    raw_docs = load_documents(settings.resources_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunked_docs: list[Document] = []

    for doc in raw_docs:
        chunks = splitter.split_text(doc.page_content)
        for idx, chunk in enumerate(chunks):
            chunked_docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        **doc.metadata,
                        "chunk_id": idx,
                    },
                )
            )

    vectorstore = Chroma(
        collection_name=settings.collection_name,
        persist_directory=str(settings.chroma_path),
        embedding_function=get_embeddings(),
    )

    try:
        vectorstore.delete_collection()
    except Exception:
        pass

    vectorstore = Chroma(
        collection_name=settings.collection_name,
        persist_directory=str(settings.chroma_path),
        embedding_function=get_embeddings(),
    )

    if chunked_docs:
        vectorstore.add_documents(chunked_docs)

    print(f"Indexed {len(raw_docs)} documents into {settings.chroma_path}")


if __name__ == "__main__":
    main()
