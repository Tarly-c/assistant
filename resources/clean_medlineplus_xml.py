#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import re
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path


TAG_RE = re.compile(r"<[^>]+>")
BREAK_REPLACEMENTS = [
    (re.compile(r"<\s*br\s*/?\s*>", re.IGNORECASE), "\n"),
    (re.compile(r"<\s*/\s*p\s*>", re.IGNORECASE), "\n\n"),
    (re.compile(r"<\s*p(?:\s+[^>]*)?>", re.IGNORECASE), ""),
    (re.compile(r"<\s*li(?:\s+[^>]*)?>", re.IGNORECASE), "\n- "),
    (re.compile(r"<\s*/\s*li\s*>", re.IGNORECASE), ""),
    (re.compile(r"<\s*/?\s*(?:ul|ol|div|section)\b[^>]*>", re.IGNORECASE), "\n"),
]
WS_RE = re.compile(r"[ \t\r\f\v]+")
MULTI_NL_RE = re.compile(r"\n{3,}")
NIH_FOOTER_RE = re.compile(r"^NIH:\s+.*$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert MedlinePlus XML into clean Markdown files containing only title and full-summary."
    )
    parser.add_argument(
        "--xml",
        required=True,
        help="Path to MedlinePlus XML file, e.g. resources/mplus_topics_2026-04-14.xml",
    )
    parser.add_argument(
        "--out",
        default="resources",
        help="Output directory for generated .md files (default: resources)",
    )
    parser.add_argument(
        "--language",
        default="English",
        help="Only export topics matching this language attribute (default: English)",
    )
    parser.add_argument(
        "--prefix",
        default="medlineplus",
        help="Filename prefix for generated markdown files (default: medlineplus)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing generated files matching <prefix>_*.md in the output directory before writing new ones.",
    )
    parser.add_argument(
        "--keep-nih-footer",
        action="store_true",
        help="Keep lines like 'NIH: National ...' if they appear inside full-summary.",
    )
    return parser.parse_args()


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text or "untitled"


def clean_summary(raw_summary: str, keep_nih_footer: bool = False) -> str:
    if not raw_summary:
        return ""

    text = html.unescape(raw_summary)
    text = text.replace("\xa0", " ")

    for pattern, replacement in BREAK_REPLACEMENTS:
        text = pattern.sub(replacement, text)

    text = TAG_RE.sub("", text)
    text = html.unescape(text)

    lines: list[str] = []
    prev_blank = False
    for line in text.splitlines():
        line = WS_RE.sub(" ", line).strip()
        if line.startswith("-"):
            line = re.sub(r"^-+\s*", "- ", line)

        if not keep_nih_footer and NIH_FOOTER_RE.match(line):
            continue

        if not line:
            if not prev_blank:
                lines.append("")
            prev_blank = True
            continue

        lines.append(line)
        prev_blank = False

    cleaned = "\n".join(lines).strip()
    cleaned = MULTI_NL_RE.sub("\n\n", cleaned)
    return cleaned


def write_topic_markdown(
    out_dir: Path,
    prefix: str,
    topic_id: str,
    title: str,
    summary: str,
) -> Path:
    filename = f"{prefix}_{slugify(title)}_{topic_id}.md"
    path = out_dir / filename
    content = f"# {title}\n\n{summary}\n"
    path.write_text(content, encoding="utf-8")
    return path


def export_topics(
    xml_path: Path,
    out_dir: Path,
    language: str,
    prefix: str,
    clean: bool,
    keep_nih_footer: bool,
) -> tuple[int, int]:
    out_dir.mkdir(parents=True, exist_ok=True)

    if clean:
        for old_file in out_dir.glob(f"{prefix}_*.md"):
            old_file.unlink()

    created = 0
    skipped = 0

    for event, elem in ET.iterparse(xml_path, events=("end",)):
        if elem.tag != "health-topic":
            continue

        topic_language = (elem.attrib.get("language") or "").strip()
        if language and topic_language.lower() != language.lower():
            elem.clear()
            continue

        topic_id = (elem.attrib.get("id") or "").strip()
        title = (elem.attrib.get("title") or "").strip()
        raw_summary = (elem.findtext("full-summary") or "").strip()
        summary = clean_summary(raw_summary, keep_nih_footer=keep_nih_footer)

        if not title or not summary or not topic_id:
            skipped += 1
            elem.clear()
            continue

        write_topic_markdown(
            out_dir=out_dir,
            prefix=prefix,
            topic_id=topic_id,
            title=title,
            summary=summary,
        )
        created += 1
        elem.clear()

    return created, skipped


def main() -> None:
    args = parse_args()
    xml_path = Path(args.xml)
    out_dir = Path(args.out)

    if not xml_path.exists():
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    created, skipped = export_topics(
        xml_path=xml_path,
        out_dir=out_dir,
        language=args.language,
        prefix=args.prefix,
        clean=args.clean,
        keep_nih_footer=args.keep_nih_footer,
    )

    print(f"Done. Created {created} markdown files in: {out_dir}")
    if skipped:
        print(f"Skipped {skipped} topics with missing title/summary/id.")


if __name__ == "__main__":
    main()
