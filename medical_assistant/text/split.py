"""纯文本处理：子句切分、观察窗口。无字符串匹配。"""
from __future__ import annotations
import re
from typing import Iterable, Sequence


def norm(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").strip().lower())


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).strip(" ，,。.;；、")


def dedupe(items: Iterable[str]) -> list[str]:
    out, seen = [], set()
    for item in items:
        c = clean(item)
        k = norm(c)
        if c and k not in seen:
            out.append(c); seen.add(k)
    return out


def split_clauses(text: str, min_len: int = 2) -> list[str]:
    """按中文标点切子句。"""
    clauses = []
    for part in re.split(r"[。；;！!？?\n]+", text or ""):
        for sub in re.split(r"[，,、]+", part):
            c = clean(sub)
            if c and len(norm(c)) >= min_len:
                clauses.append(c)
    return clauses


def split_windows(
    title: str = "", description: str = "", *,
    include_title: bool = False, extra: Sequence[str] | None = None,
) -> list[str]:
    """多粒度观察窗口（单子句 + 相邻 2/3 子句）。"""
    raw = "\n".join(p for p in [description, *(extra or [])] if p)
    clauses = split_clauses(raw, min_len=4)
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
