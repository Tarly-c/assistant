from __future__ import annotations

"""Build a generic case-level vector index from cases_demo.json.

Each JSON item is indexed as one case document. The current demo dataset happens

to contain 100 toothache cases, but the script and collection names stay generic.

Usage:
    python scripts/build_case_index.py
    MEDICAL_ASSISTANT_CASE_DATA_FILE=data/cases_demo.json python scripts/build_case_index.py
"""

import argparse
import json
import sys
from pathlib import Path

# Allow running the script from the repository root without installing the pkg.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from medical_assistant.config import get_settings
from medical_assistant.services.cases.store import case_document_text, load_cases


def build_chroma_index() -> int:
    settings = get_settings()
    cases = load_cases()
    settings.chroma_path.mkdir(parents=True, exist_ok=True)

    try:
        from langchain_chroma import Chroma
        from langchain_core.documents import Document
        from langchain_ollama import OllamaEmbeddings
    except Exception as exc:
        preview_path = settings.chroma_path / "case_index_preview.json"
        preview_path.write_text(
            json.dumps([c.model_dump() for c in cases], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print("Chroma/LangChain dependencies are not installed; wrote preview only:")
        print(f"  {preview_path}")
        print(f"Import error: {exc}")
        return 0

    embeddings = OllamaEmbeddings(model=settings.embedding_model)

    # Recreate collection for deterministic demo builds.
    try:
        old = Chroma(
            collection_name=settings.case_collection_name,
            persist_directory=str(settings.chroma_path),
            embedding_function=embeddings,
        )
        old.delete_collection()
    except Exception:
        pass

    vectorstore = Chroma(
        collection_name=settings.case_collection_name,
        persist_directory=str(settings.chroma_path),
        embedding_function=embeddings,
    )

    docs = []
    ids = []
    for case in cases:
        docs.append(
            Document(
                page_content=case_document_text(case),
                metadata={
                    "case_id": case.case_id,
                    "title": case.title,
                    "treat": case.treat,
                    "feature_tags": ",".join(case.feature_tags),
                },
            )
        )
        ids.append(case.case_id)

    vectorstore.add_documents(docs, ids=ids)
    print(
        f"Built case vector index: collection={settings.case_collection_name}, "
        f"count={len(cases)}, path={settings.chroma_path}"
    )
    return len(cases)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="only parse and print case count")
    args = parser.parse_args()

    cases = load_cases()
    if args.check:
        print(f"Loaded {len(cases)} cases from {get_settings().case_data_path}")
        print(f"First case: {cases[0].case_id} {cases[0].title}")
        return

    build_chroma_index()


if __name__ == "__main__":
    main()
