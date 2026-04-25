# Medical Assistant Case Demo

This version turns the original RAG-style workflow into a generic local case-localization workflow.

The current demo data is `data/cases_demo.json` with 100 toothache cases. File and module names are intentionally generic: future datasets can be swapped in without renaming workflow nodes.

## Workflow

```text
normalize
→ update_memory
→ narrow_cases
→ retrieve_cases
→ route_after_cases
   ├─ plan_question
   └─ final_answer
```

- `normalize`: optional LLM normalization plus deterministic keyword extraction.
- `update_memory`: stores confirmed/denied/uncertain features and binds short replies like “是/不是” to the last asked feature.
- `narrow_cases`: shrinks the local solution set with structured memory.
- `retrieve_cases`: ranks case-level records, not text chunks.
- `plan_question`: chooses the most discriminative feature question from the remaining candidates.
- `final_answer`: outputs the matched case title and the JSON `treat` field.

## Run

```bash
python scripts/build_case_index.py --check
python run.py cli --debug
```

The workflow can run without LangGraph installed by using a sequential fallback. If you have the original dependencies installed, `build_workflow()` will compile the LangGraph version automatically.

## Optional vector index

```bash
python scripts/build_case_index.py
```

This indexes each case as one Chroma document in collection `case_demo`. If Chroma/LangChain/Ollama dependencies are unavailable, the script writes `chroma_db/case_index_preview.json` so you can still inspect the case records.

## Environment options

```bash
MEDICAL_ASSISTANT_CASE_DATA_FILE=data/cases_demo.json
MEDICAL_ASSISTANT_CASE_COLLECTION_NAME=case_demo
MEDICAL_ASSISTANT_USE_LLM_NORMALIZE=false
MEDICAL_ASSISTANT_MAX_CLARIFY_TURNS=6
```
