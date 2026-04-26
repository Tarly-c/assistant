from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medical_assistant.config import get_settings
from medical_assistant.cases.store import case_document_text, load_cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    cases = load_cases()
    if args.check:
        print(f"Loaded {len(cases)} cases from {settings.case_data_path}")
        print(f"First: {cases[0].case_id} {cases[0].title}")
        return

    settings.chroma_path.mkdir(parents=True, exist_ok=True)
    try:
        from langchain_chroma import Chroma
        from langchain_core.documents import Document
        from langchain_ollama import OllamaEmbeddings
    except Exception as exc:
        print(f"Chroma deps not installed: {exc}")
        return

    emb = OllamaEmbeddings(model=settings.embedding_model)
    try:
        Chroma(collection_name=settings.case_collection_name,
               persist_directory=str(settings.chroma_path),
               embedding_function=emb).delete_collection()
    except Exception:
        pass

    vs = Chroma(collection_name=settings.case_collection_name,
                persist_directory=str(settings.chroma_path), embedding_function=emb)
    docs = [Document(page_content=case_document_text(c),
                     metadata={"case_id": c.case_id, "title": c.title, "treat": c.treat,
                               "feature_tags": ",".join(c.feature_tags)})
            for c in cases]
    vs.add_documents(docs, ids=[c.case_id for c in cases])
    print(f"Built index: {settings.case_collection_name}, {len(cases)} docs")


if __name__ == "__main__":
    main()
