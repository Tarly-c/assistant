from __future__ import annotations

import datetime as dt
import re
import xml.etree.ElementTree as ET

import requests

from medical_assistant.config import get_settings
from medical_assistant.schemas.retrieval import PubMedHit, PubMedSearchResult

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def _base_params() -> dict[str, str]:
    settings = get_settings()
    params = {"tool": settings.pubmed_tool_name}
    if settings.pubmed_email:
        params["email"] = settings.pubmed_email
    if settings.pubmed_api_key:
        params["api_key"] = settings.pubmed_api_key
    return params


def _esearch(query: str) -> list[str]:
    settings = get_settings()
    params = {
        "db": "pubmed",
        "retmode": "json",
        "sort": "relevance",
        "retmax": str(settings.pubmed_retmax),
        "term": query,
        **_base_params(),
    }
    try:
        response = requests.get(ESEARCH_URL, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        return data.get("esearchresult", {}).get("idlist", []) or []
    except Exception:
        return []


def _efetch(pmids: list[str]) -> str:
    if not pmids:
        return ""
    params = {
        "db": "pubmed",
        "retmode": "xml",
        "id": ",".join(pmids),
        **_base_params(),
    }
    try:
        response = requests.get(EFETCH_URL, params=params, timeout=20)
        response.raise_for_status()
        return response.text
    except Exception:
        return ""


def _safe_text(element) -> str:
    if element is None:
        return ""
    return "".join(element.itertext()).strip()


def _extract_pubdate(article) -> str:
    year = _safe_text(article.find(".//PubDate/Year"))
    month = _safe_text(article.find(".//PubDate/Month"))
    medline_date = _safe_text(article.find(".//PubDate/MedlineDate"))
    if year:
        return f"{year}-{month}" if month else year
    return medline_date or ""


def parse_pubmed_xml(xml_text: str) -> list[PubMedHit]:
    if not xml_text.strip():
        return []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    hits: list[PubMedHit] = []
    for article in root.findall(".//PubmedArticle"):
        pmid = _safe_text(article.find(".//PMID"))
        title = _safe_text(article.find(".//ArticleTitle"))
        journal = _safe_text(article.find(".//Journal/Title"))
        pubdate = _extract_pubdate(article)

        abstract_parts: list[str] = []
        for node in article.findall(".//Abstract/AbstractText"):
            label = node.attrib.get("Label")
            text = _safe_text(node)
            abstract_parts.append(f"{label}: {text}" if label else text)
        snippet = " ".join(part for part in abstract_parts if part).strip()[:1200]

        publication_types = [
            _safe_text(node)
            for node in article.findall(".//PublicationType")
            if _safe_text(node)
        ]
        mesh_headings = [
            _safe_text(node)
            for node in article.findall(".//MeshHeading/DescriptorName")
            if _safe_text(node)
        ]

        hits.append(
            PubMedHit(
                pmid=pmid,
                title=title,
                journal=journal,
                pubdate=pubdate,
                snippet=snippet,
                publication_types=publication_types,
                mesh_headings=mesh_headings,
                rerank_score=0.0,
            )
        )
    return hits


def _tokenize(text: str) -> list[str]:
    raw = re.split(r"[^a-zA-Z0-9]+", text.lower())
    stop = {
        "the",
        "and",
        "for",
        "with",
        "what",
        "when",
        "how",
        "why",
        "from",
        "into",
        "your",
        "have",
        "does",
        "this",
        "that",
        "guideline",
        "review",
        "treatment",
        "symptom",
        "general",
    }
    return [item for item in raw if len(item) >= 3 and item not in stop]


def _coverage_score(text: str, terms: list[str]) -> float:
    if not terms:
        return 0.0
    haystack = text.lower()
    hits = sum(1 for term in terms if term in haystack)
    return hits / max(1, len(terms))


def _evidence_type_score(publication_types: list[str]) -> float:
    joined = " | ".join(publication_types).lower()
    if "guideline" in joined or "practice guideline" in joined:
        return 1.8
    if "systematic review" in joined or "meta-analysis" in joined:
        return 1.6
    if "review" in joined:
        return 1.1
    if "randomized controlled trial" in joined:
        return 0.8
    if "case reports" in joined:
        return -0.8
    return 0.0


def _recency_score(pubdate: str) -> float:
    match = re.search(r"(19|20)\d{2}", pubdate or "")
    if not match:
        return 0.0
    year = int(match.group(0))
    current_year = dt.datetime.utcnow().year
    delta = current_year - year
    if delta <= 5:
        return 0.4
    if delta <= 10:
        return 0.2
    return 0.0


def _score_hit(question: str, query: str, hit: PubMedHit) -> float:
    terms = _tokenize(f"{question} {query}")
    title_text = hit.title or ""
    abstract_text = hit.snippet or ""
    mesh_text = " ".join(hit.mesh_headings)

    score = 0.0
    score += 1.4 * _coverage_score(title_text, terms)
    score += 0.9 * _coverage_score(abstract_text, terms)
    score += 0.5 * _coverage_score(mesh_text, terms)
    score += _evidence_type_score(hit.publication_types)
    score += _recency_score(hit.pubdate)
    return round(score, 4)


def search_pubmed_best(question: str, queries: list[str]) -> dict:
    settings = get_settings()
    dedup: dict[str, PubMedHit] = {}
    best_query = queries[0] if queries else ""

    for query in queries[:3]:
        pmids = _esearch(query)
        xml_text = _efetch(pmids)
        hits = parse_pubmed_xml(xml_text)
        for hit in hits:
            hit.rerank_score = _score_hit(question, query, hit)
            previous = dedup.get(hit.pmid)
            if previous is None or hit.rerank_score > previous.rerank_score:
                dedup[hit.pmid] = hit
        best_query = query

    ranked_hits = sorted(dedup.values(), key=lambda x: x.rerank_score, reverse=True)
    top_score = ranked_hits[0].rerank_score if ranked_hits else 0.0

    result = PubMedSearchResult(
        query=best_query,
        enough=top_score >= settings.pubmed_min_score,
        score=top_score,
        reason="generic evidence reranking by concept coverage, evidence type and recency",
        hits=ranked_hits[:3],
    )
    return result.model_dump(mode="json")
