from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medical_assistant.config import get_settings
from medical_assistant.cases.store import document_text, load_cases


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    args = p.parse_args()

    cfg = get_settings()
    cases = load_cases()
    if args.check:
        print(f"{len(cases)} cases from {cfg.case_data_path}")
        print(f"First: {cases[0].case_id} {cases[0].title}")
        return

    cfg.chroma_path.mkdir(parents=True, exist_ok=True)
    try:
        from langchain_chroma import Chroma
        from langchain_core.documents import Document
        from langchain_ollama import OllamaEmbeddings
    except Exception as e:
        print(f"Chroma deps not installed: {e}"); return

    emb = OllamaEmbeddings(model=cfg.embedding_model)
    try:
        Chroma(collection_name=cfg.case_collection_name,
               persist_directory=str(cfg.chroma_path),
               embedding_function=emb).delete_collection()
    except Exception:
        pass

    vs = Chroma(collection_name=cfg.case_collection_name,
                persist_directory=str(cfg.chroma_path), embedding_function=emb)
    docs = [Document(page_content=document_text(c),
                     metadata={"case_id": c.case_id, "title": c.title, "treat": c.treat})
            for c in cases]
    vs.add_documents(docs, ids=[c.case_id for c in cases])
    print(f"Built index: {cfg.case_collection_name}, {len(cases)} docs")


if __name__ == "__main__":
    main()
