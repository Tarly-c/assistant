# scripts

This folder has been simplified around a single build entry point:

- `rebuild_medical_resources.py`

The old one-off scripts are intended to be removed from the repo because their responsibilities are now folded into the rebuild script or they are tied to the older terminology-first retrieval flow:

- `build_index.py`
- `import_mesh.py`
- `evaluate_retrieval.py`

## What the rebuild script now does

- Parses MedlinePlus topic XML as the primary retrieval source
- Treats MeSH descriptor XML as optional reference data
- Writes topic-level structured records to `topic_cards.jsonl`
- Writes retrieval chunks to `retrieval_chunks.jsonl`
- Writes full topic markdown documents to `topic_docs/`
- Writes retrieval-unit markdown files to `retrieval_units/` for compatibility
- Optionally builds the Chroma index directly

## Example

```bash
python scripts/rebuild_medical_resources.py --topics-xml resources/raw/mplus_topics_2026-04-14.xml  --mesh-desc-xml resources/raw/desc2026.xml   --build-index
```
