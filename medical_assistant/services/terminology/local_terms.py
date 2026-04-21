from __future__ import annotations

import re

from medical_assistant.services.terminology.aliases import (
    CHIEF_COMPLAINT_PATTERNS,
    FACET_PATTERNS,
    FINDING_PATTERNS,
    NEGATION_PREFIXES,
    RED_FLAG_PATTERNS,
)


def _contains_any(text: str, patterns: list[str]) -> bool:
    return any(pattern in text for pattern in patterns)


def _is_negated(text: str, patterns: list[str]) -> bool:
    for pattern in patterns:
        for neg in NEGATION_PREFIXES:
            if re.search(rf"{re.escape(neg)}.{{0,4}}{re.escape(pattern)}", text):
                return True
    return False


def detect_chief_complaint(text: str) -> dict | None:
    for item in CHIEF_COMPLAINT_PATTERNS:
        if _contains_any(text, item["patterns"]):
            return item
    return None


def detect_findings(text: str) -> list[dict]:
    results: list[dict] = []
    for item in FINDING_PATTERNS + FACET_PATTERNS:
        if _contains_any(text, item["patterns"]) and not _is_negated(text, item["patterns"]):
            results.append(item)
    return results


def detect_negated_findings(text: str) -> list[dict]:
    results: list[dict] = []
    for item in FINDING_PATTERNS + FACET_PATTERNS:
        if _contains_any(text, item["patterns"]) and _is_negated(text, item["patterns"]):
            results.append(item)
    return results


def detect_red_flags(text: str) -> list[str]:
    results: list[str] = []
    for item in RED_FLAG_PATTERNS:
        if _contains_any(text, item["patterns"]) and not _is_negated(text, item["patterns"]):
            results.append(item["name"])
    return results
