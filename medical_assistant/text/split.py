"""纯文本处理：子句切分、观察窗口、搜索词提取。"""
from __future__ import annotations

import re
from typing import Iterable, Sequence


def norm(text: str) -> str:
    """去空格、小写化。"""
    return re.sub(r"\s+", "", (text or "").strip().lower())


def clean(text: str) -> str:
    """清理首尾标点和多余空格。"""
    return re.sub(r"\s+", " ", (text or "").strip()).strip(" ，,。.;；、")


def dedupe(items: Iterable[str]) -> list[str]:
    """保序去重。"""
    out, seen = [], set()
    for item in items:
        c = clean(item)
        k = norm(c)
        if c and k not in seen:
            out.append(c)
            seen.add(k)
    return out


def split_clauses(text: str) -> list[str]:
    """按中文标点切子句（≥4 字）。"""
    clauses = []
    for part in re.split(r"[。；;！!？?\n]+", text or ""):
        for sub in re.split(r"[，,、]+", part):
            c = clean(sub)
            if c and len(norm(c)) >= 4:
                clauses.append(c)
    return clauses


def split_windows(
    title: str = "", description: str = "", *,
    include_title: bool = False, extra: Sequence[str] | None = None,
) -> list[str]:
    """多粒度观察窗口（单子句 + 相邻 2/3 子句拼接）。"""
    raw = "\n".join(p for p in [description, *(extra or [])] if p)
    clauses = split_clauses(raw)
    units = []
    if include_title and title:
        units.append(clean(title))
    for i in range(len(clauses)):
        for w in (1, 2, 3):
            if i + w > len(clauses):
                break
            window = clean("，".join(clauses[i: i + w]))
            if 4 <= len(norm(window)) <= 180:
                units.append(window)
    return dedupe(units)


def extract_terms(text: str) -> list[str]:
    """从用户输入中提取搜索短语和子片段。"""
    text = clean(text)
    if not text:
        return []
    terms = []
    for chunk in split_clauses(text):
        terms.append(chunk)
        c = norm(chunk)
        if len(c) >= 4:
            for n in (2, 3):
                for i in range(min(len(c) - n + 1, 6)):
                    terms.append(c[i: i + n])
        terms.extend(re.findall(r"[a-z][a-z0-9_/-]{2,}", chunk.lower()))
    return dedupe(terms)[:16]
