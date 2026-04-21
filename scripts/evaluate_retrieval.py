from __future__ import annotations

from medical_assistant.services.retrieval.local import search_local


QUERIES = [
    ("头痛恶心", "headache nausea"),
    ("咳嗽气短", "cough shortness of breath"),
    ("发热喉咙痛", "fever sore throat"),
]


def main() -> None:
    for zh_query, en_query in QUERIES:
        result = search_local(
            question=zh_query,
            local_query=en_query,
            normalized_terms=en_query.split(),
        )
        print("=" * 80)
        print("question:", zh_query)
        print("local_query:", en_query)
        print("score:", result["score"])
        print("enough:", result["enough"])
        for idx, hit in enumerate(result["hits"][:3], start=1):
            print(f"{idx}. {hit['source']} | score={hit['score']}")
            print(hit["snippet"][:200])
            print()


if __name__ == "__main__":
    main()
