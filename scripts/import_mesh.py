from __future__ import annotations

import json

from medical_assistant.config import get_settings
from medical_assistant.services.terminology.mesh import lookup_mesh


SEED_TERMS = [
    "headache",
    "migraine",
    "fever",
    "cough",
    "sore throat",
    "abdominal pain",
    "diarrhea",
    "shortness of breath",
    "chest pain",
    "rash",
    "nausea",
    "vomiting",
]


def main() -> None:
    settings = get_settings()
    settings.mesh_cache_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    seen: set[str] = set()

    for term in SEED_TERMS:
        for item in lookup_mesh(term, limit=5):
            key = str(item.get("id") or item.get("label"))
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "id": item.get("id"),
                    "label": item.get("label"),
                    "aliases": item.get("aliases", []),
                }
            )

    with settings.mesh_cache_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved {len(rows)} MeSH rows to {settings.mesh_cache_path}")


if __name__ == "__main__":
    main()
