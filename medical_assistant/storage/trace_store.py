from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from medical_assistant.config import get_settings


class JsonlTraceStore:
    def __init__(self, trace_dir: str | None = None) -> None:
        settings = get_settings()
        self.trace_dir = Path(trace_dir or settings.trace_path)
        self.trace_dir.mkdir(parents=True, exist_ok=True)

    def write(self, session_id: str, payload: dict[str, Any]) -> None:
        path = self.trace_dir / f"{session_id}.jsonl"
        record = {
            "timestamp": time.time(),
            **payload,
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
