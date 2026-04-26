# No-wordlist decision-tree patch

This patch removes the remaining hand-written example lists from the decision-tree/probe path.

Apply from your repo root:

```powershell
Copy-Item -Recurse -Force .\assistant_no_wordlist_patch\* D:\Project_py\assistant\
python scripts/build_question_tree.py --print-tree
python run.py cli --debug
```

LLM answer parsing is no longer hidden behind rule lists. To inspect the exact payload before invoking a model:

```powershell
python scripts/debug_answer_parser_payload.py --question "你的牙痛是不是遇冷明显？" --answer "不是，我是晚上自己疼"
```

To invoke explicitly:

```powershell
$env:MEDICAL_ASSISTANT_DEBUG_LLM_PAYLOADS="true"
python scripts/debug_answer_parser_payload.py --question "你的牙痛是不是遇冷明显？" --answer "不是，我是晚上自己疼" --invoke
```

Files changed:

- `medical_assistant/services/cases/features.py`
- `medical_assistant/services/cases/answer_parser.py`
- `medical_assistant/services/cases/memory.py`
- `medical_assistant/schemas.py`
- `medical_assistant/prompts.py`
- `medical_assistant/llm.py`
- `medical_assistant/config.py`
- `medical_assistant/graph/nodes/normalize.py`
- `scripts/debug_answer_parser_payload.py`
