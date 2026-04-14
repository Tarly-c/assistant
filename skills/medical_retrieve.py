# skills/medical_retrieve.py
from pathlib import Path
from pydantic import BaseModel, Field
import os
import re
import requests

ROOT = Path(__file__).resolve().parent.parent
RESOURCE_DIR = ROOT / "resources"

def tokenize(text: str):
    return re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9]+", text.lower())

def local_search(query: str, top_k: int = 3):
    if not RESOURCE_DIR.exists():
        return {"source": "local", "enough": False, "score": 0.0, "hits": [], "reason": "NO_LOCAL_DB"}

    q = set(tokenize(query))
    hits = []

    for path in RESOURCE_DIR.rglob("*"):
        if path.suffix.lower() not in {".txt", ".md"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        tokens = set(tokenize(text))
        if not tokens:
            continue

        score = len(q & tokens) / max(len(q), 1)
        if score > 0:
            snippet = text[:500].replace("\n", " ")
            hits.append({
                "title": path.name,
                "file": str(path),
                "snippet": snippet,
                "score": round(score, 3),
            })

    hits.sort(key=lambda x: x["score"], reverse=True)
    best = hits[0]["score"] if hits else 0.0
    enough = best >= 0.5
    return {"source": "local", "enough": enough, "score": best, "hits": hits[:top_k], "reason": None if enough else "LOW_CONFIDENCE"}

def pubmed_search(query: str, retmax: int = 5):
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    common = {
        "db": "pubmed",
        "tool": "tarly_assistant",
        "email": os.getenv("NCBI_EMAIL", "your_email@example.com"),
    }

    r = requests.get(
        f"{base}/esearch.fcgi",
        params={**common, "retmode": "json", "sort": "relevance", "retmax": retmax, "term": query},
        timeout=10,
    )
    ids = r.json().get("esearchresult", {}).get("idlist", [])
    if not ids:
        return {"source": "pubmed", "enough": False, "score": 0.0, "hits": [], "reason": "NO_WEB_HIT"}

    s = requests.get(
        f"{base}/esummary.fcgi",
        params={**common, "retmode": "json", "id": ",".join(ids)},
        timeout=10,
    )
    data = s.json().get("result", {})

    hits = []
    for pmid in ids:
        item = data.get(pmid, {})
        hits.append({
            "pmid": pmid,
            "title": item.get("title"),
            "pubdate": item.get("pubdate"),
            "source": "PubMed",
        })

    return {"source": "pubmed", "enough": bool(hits), "score": 1.0 if hits else 0.0, "hits": hits, "reason": None if hits else "NO_WEB_HIT"}

def medical_retrieve(query: str):
    local = local_search(query)
    if local["enough"]:
        return local
    return pubmed_search(query)

class MedicalRetrieveArgs(BaseModel):
    query: str = Field(..., description="医学问题或检索关键词")

SKILL_DEF = {
    "name": "medical_retrieve",
    "description": "医学检索工具。必须先查本地知识库；本地结果不足时自动回退到 PubMed。",
    "func": medical_retrieve,
    "args_schema": MedicalRetrieveArgs,
}
