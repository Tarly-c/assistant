from __future__ import annotations

from pathlib import Path

from medical_assistant.schemas.response import EvidenceItem
from medical_assistant.schemas.retrieval import CandidateTopic, LocalSearchResult, PubMedSearchResult


def render_candidate_topics_for_user(candidate_topics: list[CandidateTopic]) -> str:
    if not candidate_topics:
        return "当前还没有足够稳定的匹配条目。"

    lines: list[str] = []
    for idx, item in enumerate(candidate_topics[:3], start=1):
        matched = "、".join(item.matched_terms) if item.matched_terms else "部分相关"
        missing = "、".join(item.missing_terms) if item.missing_terms else "暂无"
        lines.append(f"{idx}. {item.title}（匹配词：{matched}；待确认：{missing}）")
    return "\n".join(lines)


def render_evidence_block(local_result: LocalSearchResult, web_result: PubMedSearchResult) -> str:
    parts: list[str] = []

    for hit in local_result.hits:
        parts.append(
            f"[Local] file={Path(hit.source).name} chunk={hit.chunk_id} score={hit.score}\n{hit.snippet}"
        )

    for hit in web_result.hits:
        parts.append(
            f"[PubMed] PMID={hit.pmid} title={hit.title} journal={hit.journal} "
            f"pubdate={hit.pubdate} score={hit.rerank_score}\n{hit.snippet}"
        )

    return "\n\n".join(parts) if parts else "无可用证据。"


def render_sources_text(local_result: LocalSearchResult, web_result: PubMedSearchResult) -> str:
    lines: list[str] = []
    seen: set[str] = set()

    for hit in local_result.hits:
        label = f"- {Path(hit.source).name} (chunk {hit.chunk_id})"
        if label not in seen:
            seen.add(label)
            lines.append(label)

    for hit in web_result.hits:
        label = f"- PMID {hit.pmid}: {hit.title}"
        if label not in seen:
            seen.add(label)
            lines.append(label)

    return "\n".join(lines) if lines else "- 暂无来源"


def build_evidence_items(local_result: LocalSearchResult, web_result: PubMedSearchResult) -> list[EvidenceItem]:
    items: list[EvidenceItem] = []

    for hit in local_result.hits:
        items.append(
            EvidenceItem(
                source_type="local",
                label=Path(hit.source).name,
                title=hit.title or Path(hit.source).stem,
                detail=f"chunk {hit.chunk_id}",
            )
        )

    for hit in web_result.hits:
        items.append(
            EvidenceItem(
                source_type="pubmed",
                label=f"PMID {hit.pmid}",
                title=hit.title,
                detail=hit.journal,
            )
        )

    return items
