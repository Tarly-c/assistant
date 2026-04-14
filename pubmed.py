from __future__ import annotations

import os
import re
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List

import requests

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
NCBI_TOOL = os.getenv("NCBI_TOOL", "medical_langgraph_assistant")
NCBI_EMAIL = os.getenv("NCBI_EMAIL", "")
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")


def _clean(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _common_params() -> Dict[str, str]:
    params: Dict[str, str] = {"tool": NCBI_TOOL}
    if NCBI_EMAIL:
        params["email"] = NCBI_EMAIL
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    return params


def esearch(term: str, retmax: int = 8) -> List[str]:
    resp = requests.get(
        f"{BASE_URL}/esearch.fcgi",
        params={
            **_common_params(),
            "db": "pubmed",
            "retmode": "json",
            "retmax": retmax,
            "sort": "relevance",
            "term": term,
        },
        timeout=20,
    )
    resp.raise_for_status()
    return resp.json().get("esearchresult", {}).get("idlist", [])


def efetch(pmids: List[str]) -> str:
    resp = requests.get(
        f"{BASE_URL}/efetch.fcgi",
        params={
            **_common_params(),
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
        },
        timeout=20,
    )
    resp.raise_for_status()
    return resp.text


def parse_pubmed_xml(xml_text: str) -> List[Dict[str, Any]]:
    root = ET.fromstring(xml_text)
    hits: List[Dict[str, Any]] = []

    for article in root.findall(".//PubmedArticle"):
        pmid = _clean(article.findtext(".//MedlineCitation/PMID", default=""))

        title_node = article.find(".//ArticleTitle")
        title = _clean("".join(title_node.itertext()) if title_node is not None else "")

        journal = _clean(article.findtext(".//Journal/Title", default=""))
        pubdate = (
            _clean(article.findtext(".//JournalIssue/PubDate/Year", default=""))
            or _clean(article.findtext(".//JournalIssue/PubDate/MedlineDate", default=""))
        )

        abstract_parts: List[str] = []
        for node in article.findall(".//Abstract/AbstractText"):
            label = _clean(node.attrib.get("Label", ""))
            text = _clean("".join(node.itertext()))
            if not text:
                continue
            abstract_parts.append(f"{label}: {text}" if label else text)

        snippet = _clean(" ".join(abstract_parts))

        if title or snippet:
            hits.append(
                {
                    "pmid": pmid,
                    "title": title,
                    "journal": journal,
                    "pubdate": pubdate,
                    "snippet": snippet[:1500],
                }
            )

    return hits


def _question_hints(question: str) -> tuple[list[str], list[str]]:
    q = question.lower()
    positives: list[str] = []
    negatives: list[str] = []

    if "头痛" in question or "headache" in q:
        positives += [
            "headache",
            "migraine",
            "tension-type",
            "acute",
            "treatment",
            "therapy",
            "management",
            "guideline",
            "review",
        ]
        negatives += [
            "medication overuse headache",
            "overuse headache",
        ]

    if "发热" in question or "发烧" in question or "fever" in q:
        positives += ["fever", "management", "treatment", "guideline", "review"]

    if "咳嗽" in question or "cough" in q:
        positives += ["cough", "management", "treatment", "guideline", "review"]

    if any(x in question for x in ["吃什么药", "用什么药", "怎么办"]) or any(
        x in q for x in ["treatment", "therapy", "management", "medication", "drug"]
    ):
        positives += ["treatment", "therapy", "management", "guideline", "review"]

    return positives, negatives


def rerank_hits(question: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    positives, negatives = _question_hints(question)
    rescored: List[Dict[str, Any]] = []

    for hit in hits:
        text = f"{hit.get('title', '')} {hit.get('snippet', '')}".lower()
        score = 0.0

        for term in positives:
            if term in text:
                score += 1.0

        for term in negatives:
            if term in text:
                score -= 3.0

        rescored.append({**hit, "rerank_score": round(score, 2)})

    rescored.sort(key=lambda x: x["rerank_score"], reverse=True)
    return rescored


def search_pubmed_best(
    question: str,
    queries: List[str],
    retmax: int = 8,
    min_score: float = 2.0,
) -> dict:
    queries = [q.strip() for q in queries if q and q.strip()]
    debug_errors: List[str] = []
    tried: List[str] = []

    best_payload: dict | None = None
    best_score = float("-inf")

    for query in queries:
        tried.append(query)

        try:
            pmids = esearch(query, retmax=retmax)
            if not pmids:
                continue

            time.sleep(0.34)
            xml_text = efetch(pmids)
            hits = parse_pubmed_xml(xml_text)
            if not hits:
                continue

            hits = rerank_hits(question, hits)
            top_score = hits[0]["rerank_score"]

            if top_score > best_score:
                best_score = top_score
                best_payload = {
                    "query_used": query,
                    "hits": hits[:3],
                }

        except Exception as exc:
            debug_errors.append(f"{query}: {exc}")

        time.sleep(0.34)

    if not best_payload:
        return {
            "enough": False,
            "score": 0.0,
            "query_used": None,
            "queries_tried": tried,
            "hits": [],
            "reason": "NO_WEB_HIT" if not debug_errors else "PUBMED_REQUEST_FAILED",
            "debug": {"errors": debug_errors},
        }

    enough = best_score >= min_score
    return {
        "enough": enough,
        "score": best_score,
        "query_used": best_payload["query_used"],
        "queries_tried": tried,
        "hits": best_payload["hits"],
        "reason": None if enough else "LOW_RELEVANCE_WEB_HIT",
        "debug": {"errors": debug_errors},
    }
