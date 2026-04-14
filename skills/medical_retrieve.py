import json
import os
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any

from openai import OpenAI
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
RESOURCE_DIR = ROOT / "resources"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

NCBI_TOOL = os.getenv("NCBI_TOOL", "assistant_medical_agent")
NCBI_EMAIL = os.getenv("NCBI_EMAIL", "replace_me@example.com")
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")

llm = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

ZH_EN_TERM_MAP = {
    "头痛": "headache",
    "偏头痛": "migraine",
    "发热": "fever",
    "发烧": "fever",
    "咳嗽": "cough",
    "腹泻": "diarrhea",
    "恶心": "nausea",
    "呕吐": "vomiting",
    "高血压": "hypertension",
    "糖尿病": "diabetes",
    "胸痛": "chest pain",
    "胃痛": "abdominal pain",
    "腹痛": "abdominal pain",
    "关节痛": "joint pain",
    "喉咙痛": "sore throat",
    "咽痛": "sore throat",
    "鼻塞": "nasal congestion",
    "失眠": "insomnia",
}

INTENT_MAP = {
    "吃什么药": "treatment",
    "用什么药": "treatment",
    "什么药": "treatment",
    "治疗": "treatment",
    "怎么治": "treatment",
    "原因": "cause",
    "症状": "symptoms",
    "检查": "diagnosis",
    "怎么办": "management",
}

def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()

def _tokenize(text: str) -> List[str]:
    return re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9]+", (text or "").lower())

def _chunk_text(text: str, max_chars: int = 900, overlap: int = 150) -> List[str]:
    text = text or ""
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks

def _safe_json_loads(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match:
            return json.loads(match.group(0))
        raise

def _http_get_json(url: str, params: Dict[str, Any], timeout: int = 12):
    params = {k: v for k, v in params.items() if v not in (None, "", [])}
    full_url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(
        full_url,
        headers={"User-Agent": f"{NCBI_TOOL}/1.0 ({NCBI_EMAIL})"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8")), full_url

def _http_get_text(url: str, params: Dict[str, Any], timeout: int = 12):
    params = {k: v for k, v in params.items() if v not in (None, "", [])}
    full_url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(
        full_url,
        headers={"User-Agent": f"{NCBI_TOOL}/1.0 ({NCBI_EMAIL})"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="ignore"), full_url

def heuristic_pubmed_queries(user_query: str) -> List[str]:
    detected_terms = []
    for zh, en in ZH_EN_TERM_MAP.items():
        if zh in user_query:
            detected_terms.append(en)

    detected_terms = list(dict.fromkeys(detected_terms))
    intent = ""
    for zh, en in INTENT_MAP.items():
        if zh in user_query:
            intent = en
            break

    queries = []
    for term in detected_terms:
        if intent:
            queries.append(f"{term} {intent}")
        queries.append(term)

        if term == "headache":
            queries.extend([
                "acute headache treatment",
                "headache drug therapy adults",
                "migraine acute treatment",
                "tension-type headache treatment",
            ])

    if not queries and re.search(r"[A-Za-z]", user_query):
        queries.append(_normalize_space(user_query))

    # 去重
    seen = set()
    final_queries = []
    for q in queries:
        q = _normalize_space(q)
        if q and q not in seen:
            final_queries.append(q)
            seen.add(q)

    return final_queries[:5]

def rewrite_pubmed_queries(user_query: str) -> Dict[str, Any]:
    """
    返回:
    {
      "queries": [...],
      "debug": {"rewrite_error": "..."}  # 可选
    }
    """
    prompt = """
You rewrite a user's medical question into short English PubMed search queries.

Return JSON only:
{"queries": ["query1", "query2", "query3"]}

Requirements:
- 3 to 5 queries
- short keyword style, not full sentences
- English only
- prefer clinical concepts + treatment/cause/symptom/diagnosis intent
- include synonym/variant when useful
- do not explain
""".strip()

    llm_queries = []
    rewrite_error = None

    try:
        response = llm.chat.completions.create(
            model=OLLAMA_MODEL,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_query},
            ],
        )
        content = response.choices[0].message.content or "{}"
        data = _safe_json_loads(content)
        llm_queries = [
            _normalize_space(q)
            for q in data.get("queries", [])
            if isinstance(q, str) and _normalize_space(q)
        ]
    except Exception as e:
        rewrite_error = str(e)

    merged = []
    seen = set()

    for q in llm_queries + heuristic_pubmed_queries(user_query):
        if q and q not in seen:
            merged.append(q)
            seen.add(q)

    if not merged:
        merged = heuristic_pubmed_queries(user_query)
    if not merged:
        merged = [_normalize_space(user_query)]

    out = {"queries": merged[:5]}
    if rewrite_error:
        out["debug"] = {"rewrite_error": rewrite_error}
    return out

def local_search(query: str, top_k: int = 3, threshold: float = 0.45) -> Dict[str, Any]:
    if not RESOURCE_DIR.exists():
        return {
            "source": "local",
            "enough": False,
            "score": 0.0,
            "hits": [],
            "reason": "NO_LOCAL_DB",
        }

    q_tokens = set(_tokenize(query))
    if not q_tokens:
        return {
            "source": "local",
            "enough": False,
            "score": 0.0,
            "hits": [],
            "reason": "EMPTY_QUERY",
        }

    hits = []

    for path in RESOURCE_DIR.rglob("*"):
        if path.suffix.lower() not in {".txt", ".md"}:
            continue

        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        for idx, chunk in enumerate(_chunk_text(text)):
            c_tokens = set(_tokenize(chunk))
            if not c_tokens:
                continue

            overlap = q_tokens & c_tokens
            score = len(overlap) / max(len(q_tokens), 1)

            if query.lower() in chunk.lower():
                score += 0.35

            if score <= 0:
                continue

            snippet = _normalize_space(chunk)[:400]
            hits.append({
                "title": path.name,
                "file": str(path),
                "chunk_id": idx,
                "score": round(score, 3),
                "snippet": snippet,
            })

    hits.sort(key=lambda x: x["score"], reverse=True)
    best_score = hits[0]["score"] if hits else 0.0

    return {
        "source": "local",
        "enough": best_score >= threshold,
        "score": best_score,
        "hits": hits[:top_k],
        "reason": None if hits else "NO_LOCAL_HIT",
    }

def pubmed_esearch(term: str, retmax: int = 5):
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "retmode": "json",
        "retmax": retmax,
        "sort": "relevance",
        "term": term,
        "tool": NCBI_TOOL,
        "email": NCBI_EMAIL,
        "api_key": NCBI_API_KEY,
    }
    data, url = _http_get_json(base, params)
    ids = data.get("esearchresult", {}).get("idlist", [])
    return ids, url

def pubmed_efetch_xml(pmids: List[str]):
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "tool": NCBI_TOOL,
        "email": NCBI_EMAIL,
        "api_key": NCBI_API_KEY,
    }
    xml_text, url = _http_get_text(base, params)
    return xml_text, url

def parse_pubmed_xml(xml_text: str) -> List[Dict[str, Any]]:
    root = ET.fromstring(xml_text)
    hits = []

    for article in root.findall(".//PubmedArticle"):
        pmid = article.findtext(".//MedlineCitation/PMID", default="")
        title = _normalize_space("".join(article.findtext(".//ArticleTitle", default="")))
        journal = _normalize_space(article.findtext(".//Journal/Title", default=""))
        year = (
            article.findtext(".//JournalIssue/PubDate/Year", default="")
            or article.findtext(".//PubDate/MedlineDate", default="")
        )

        abstract_parts = []
        for node in article.findall(".//Abstract/AbstractText"):
            label = _normalize_space(node.attrib.get("Label", ""))
            text = _normalize_space("".join(node.itertext()))
            if not text:
                continue
            if label:
                abstract_parts.append(f"{label}: {text}")
            else:
                abstract_parts.append(text)

        abstract = _normalize_space(" ".join(abstract_parts))

        if title or abstract:
            hits.append({
                "pmid": pmid,
                "title": title,
                "journal": journal,
                "pubdate": year,
                "snippet": abstract[:1200],
            })

    return hits

def pubmed_search(user_query: str, retmax: int = 5) -> Dict[str, Any]:
    rewrite = rewrite_pubmed_queries(user_query)
    queries = rewrite["queries"]

    debug = {
        "queries_tried": [],
        "errors": [],
    }
    if "debug" in rewrite:
        debug.update(rewrite["debug"])

    for q in queries:
        debug["queries_tried"].append(q)

        try:
            pmids, esearch_url = pubmed_esearch(q, retmax=retmax)
        except Exception as e:
            debug["errors"].append(f"ESearch failed for '{q}': {e}")
            continue

        if not pmids:
            continue

        try:
            xml_text, efetch_url = pubmed_efetch_xml(pmids)
            hits = parse_pubmed_xml(xml_text)
        except Exception as e:
            debug["errors"].append(f"EFetch/parse failed for '{q}': {e}")
            continue

        if hits:
            return {
                "source": "pubmed",
                "enough": True,
                "score": 1.0,
                "query_used": q,
                "queries_tried": debug["queries_tried"],
                "hits": hits[:3],
                "reason": None,
                "debug": {
                    **debug,
                    "esearch_url": esearch_url,
                    "efetch_url": efetch_url,
                },
            }

    reason = "PUBMED_REQUEST_FAILED" if debug["errors"] else "NO_WEB_HIT"
    return {
        "source": "pubmed",
        "enough": False,
        "score": 0.0,
        "query_used": None,
        "queries_tried": debug["queries_tried"],
        "hits": [],
        "reason": reason,
        "debug": debug,
    }

def medical_retrieve(query: str) -> Dict[str, Any]:
    local = local_search(query)

    if local["enough"]:
        local["route"] = "local_first_hit"
        return local

    web = pubmed_search(query)
    web["route"] = "local_then_pubmed"
    web["local_diagnostic"] = {
        "score": local["score"],
        "reason": local["reason"],
        "hits": local["hits"][:2],
    }
    return web

class MedicalRetrieveArgs(BaseModel):
    query: str = Field(..., description="用户的医学问题或检索关键词")

SKILL_DEF = {
    "name": "medical_retrieve",
    "description": "医学检索工具。必须先查本地知识库；若本地证据不足，再自动查询 PubMed。",
    "func": medical_retrieve,
    "args_schema": MedicalRetrieveArgs,
}
