from __future__ import annotations

from pathlib import Path

from medical_assistant.schemas.retrieval import CandidateTopic, RetrievalHit


def aggregate_candidate_topics(
    hits: list[RetrievalHit],
    normalized_terms: list[str],
    max_topics: int = 3,
) -> list[dict]:
    groups: dict[str, dict] = {}
    normalized_terms = [term.lower() for term in normalized_terms if term]

    for hit in hits:
        key = hit.source or hit.title
        group = groups.setdefault(
            key,
            {
                "title": hit.title or Path(hit.source).stem or "local topic",
                "source": hit.source,
                "score": 0.0,
                "matched_terms": set(),
                "chunk_ids": set(),
                "reasons": [],
            },
        )

        group["score"] = max(group["score"], float(hit.score))
        if hit.chunk_id is not None:
            group["chunk_ids"].add(hit.chunk_id)

        haystack = f"{hit.title}\n{hit.snippet}".lower()
        matched_terms = [term for term in normalized_terms if term in haystack]
        for term in matched_terms:
            group["matched_terms"].add(term)

    topics: list[CandidateTopic] = []
    for item in groups.values():
        matched_terms = sorted(item["matched_terms"])
        missing_terms = [term for term in normalized_terms if term not in matched_terms]
        score = round(item["score"] + 0.03 * len(matched_terms), 4)
        reasons = []
        if matched_terms:
            reasons.append(f"命中术语：{', '.join(matched_terms)}")
        else:
            reasons.append("语义检索相关，但显式术语覆盖较少")

        topics.append(
            CandidateTopic(
                title=item["title"],
                source=item["source"],
                score=score,
                matched_terms=matched_terms[:6],
                missing_terms=missing_terms[:4],
                reasons=reasons,
                chunk_ids=sorted(item["chunk_ids"])[:5],
            )
        )

    topics.sort(key=lambda x: (x.score, len(x.matched_terms)), reverse=True)
    return [topic.model_dump(mode="json") for topic in topics[:max_topics]]


def topic_gap(candidate_topics: list[dict]) -> float:
    if len(candidate_topics) < 2:
        return 1.0
    top1 = float(candidate_topics[0].get("score", 0.0))
    top2 = float(candidate_topics[1].get("score", 0.0))
    return round(top1 - top2, 4)
