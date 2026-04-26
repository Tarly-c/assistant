"""纯文本处理：子句切分、观察窗口、搜索词提取。不含任何业务逻辑。"""
from __future__ import annotations

import re
from typing import Iterable, Sequence


def norm_text(text: str) -> str:
    """去空格、小写化。"""
    return re.sub(r"\s+", "", (text or "").strip().lower())


def readable(text: str) -> str:
    """清理首尾标点和多余空格。"""
    text = re.sub(r"\s+", " ", (text or "").strip())
    return text.strip(" ，,。.;；、")


def dedupe_keep_order(items: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        cleaned = readable(item)
        key = norm_text(cleaned)
        if cleaned and key not in seen:
            out.append(cleaned)
            seen.add(key)
    return out


def split_clauses(text: str) -> list[str]:
    """按中文标点把文本切成子句。"""
    raw = re.split(r"[。；;！!？?\n]+", text or "")
    clauses: list[str] = []
    for part in raw:
        for sub in re.split(r"[，,、]+", part):
            cleaned = readable(sub)
            if cleaned and len(norm_text(cleaned)) >= 4:
                clauses.append(cleaned)
    return clauses


def split_observation_units(
    title: str = "",
    description: str = "",
    *,
    include_title: bool = False,
    extra_texts: Sequence[str] | None = None,
) -> list[str]:
    """多粒度观察窗口（单子句 + 相邻 2/3 子句）。"""
    raw_parts = [description or ""]
    raw_parts.extend(extra_texts or [])
    raw = "\n".join(p for p in raw_parts if p)
    clauses = split_clauses(raw)

    units: list[str] = []
    if include_title and title:
        units.append(readable(title))
    for i in range(len(clauses)):
        for width in (1, 2, 3):
            if i + width > len(clauses):
                break
            window = readable("，".join(clauses[i: i + width]))
            compact = norm_text(window)
            if 4 <= len(compact) <= 180:
                units.append(window)
    return dedupe_keep_order(units)


def extract_search_terms(text: str) -> list[str]:
    """从用户输入中提取搜索用的短语和子片段。"""
    text = readable(text)
    if not text:
        return []
    chunks = split_clauses(text)
    terms: list[str] = []
    for chunk in chunks:
        terms.append(chunk)
        compact = norm_text(chunk)
        if len(compact) >= 4:
            for n in (2, 3):
                for i in range(min(len(compact) - n + 1, 6)):
                    terms.append(compact[i: i + n])
        # 英文术语
        words = re.findall(r"[a-z][a-z0-9_/-]{2,}", chunk.lower())
        terms.extend(words)
    return dedupe_keep_order(terms)[:16]
