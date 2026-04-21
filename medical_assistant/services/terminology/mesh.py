from __future__ import annotations

import json
from functools import lru_cache

import requests

from medical_assistant.config import get_settings


@lru_cache(maxsize=1)
def _load_local_mesh_cache() -> list[dict]:
    settings = get_settings()
    path = settings.mesh_cache_path
    if not path.exists():
        return []

    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _lookup_local(term: str, limit: int = 3) -> list[dict]:
    term_lower = term.lower()
    rows = _load_local_mesh_cache()
    matches: list[dict] = []
    for row in rows:
        label = str(row.get("label", "")).lower()
        aliases = [str(x).lower() for x in row.get("aliases", [])]
        if term_lower == label or term_lower in aliases or term_lower in label:
            matches.append(row)
            if len(matches) >= limit:
                break
    return matches


def _lookup_remote(term: str, limit: int = 3) -> list[dict]:
    try:
        response = requests.get(
            "https://id.nlm.nih.gov/mesh/lookup/descriptor",
            params={"label": term, "limit": limit, "match": "contains"},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
    except Exception:
        return []

    matches: list[dict] = []
    for item in data[:limit]:
        matches.append(
            {
                "id": item.get("resource"),
                "label": item.get("label"),
                "aliases": [],
            }
        )
    return matches


def lookup_mesh(term: str, limit: int = 3) -> list[dict]:
    local_hits = _lookup_local(term, limit=limit)
    if local_hits:
        return local_hits[:limit]
    return _lookup_remote(term, limit=limit)
