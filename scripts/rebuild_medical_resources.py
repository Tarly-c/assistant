from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def normalize_space(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def dedupe_keep_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        value = normalize_space(str(item))
        if not value:
            continue
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def slugify(text: str, max_len: int = 80) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text[:max_len] or "topic"


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def text_content(elem: ET.Element | None) -> str:
    if elem is None:
        return ""
    return normalize_space("".join(elem.itertext()))


def child_elements(elem: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in list(elem) if local_name(child.tag) == name]


def first_child(elem: ET.Element, name: str) -> ET.Element | None:
    for child in list(elem):
        if local_name(child.tag) == name:
            return child
    return None


def inner_xml(elem: ET.Element | None) -> str:
    if elem is None:
        return ""
    parts: list[str] = []
    if elem.text:
        parts.append(elem.text)
    for child in list(elem):
        parts.append(ET.tostring(child, encoding="unicode"))
    return "".join(parts).strip()


def summary_html_to_text(fragment: str) -> str:
    if not fragment.strip():
        return ""

    text = fragment
    text = re.sub(r"<\s*br\s*/?\s*>", "\n", text, flags=re.I)
    text = re.sub(r"</\s*p\s*>", "\n\n", text, flags=re.I)
    text = re.sub(r"<\s*p[^>]*>", "", text, flags=re.I)
    text = re.sub(r"<\s*li[^>]*>", "\n- ", text, flags=re.I)
    text = re.sub(r"</\s*(ul|ol)\s*>", "\n", text, flags=re.I)
    text = re.sub(r"<\s*(ul|ol)[^>]*>", "\n", text, flags=re.I)
    text = re.sub(r"</?\s*a[^>]*>", "", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return normalize_space(text)


def split_blocks(text: str, max_chars: int = 900) -> list[str]:
    text = normalize_space(text)
    if not text:
        return []

    blocks = [blk.strip() for blk in re.split(r"\n\s*\n", text) if blk.strip()]
    if not blocks:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if current:
            chunks.append("\n\n".join(current).strip())
        current = []
        current_len = 0

    for block in blocks:
        if len(block) > max_chars:
            lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
            for line in lines:
                if current_len and current_len + len(line) + 1 > max_chars:
                    flush()
                if len(line) > max_chars:
                    if current:
                        flush()
                    for i in range(0, len(line), max_chars):
                        chunks.append(line[i : i + max_chars].strip())
                else:
                    current.append(line)
                    current_len += len(line) + 1
            flush()
            continue

        if current_len and current_len + len(block) + 2 > max_chars:
            flush()

        current.append(block)
        current_len += len(block) + 2

    flush()
    return chunks


# ---------- MedlinePlus XML ----------

def parse_medlineplus_topics(xml_path: Path, language_filter: str | None = "English") -> list[dict]:
    rows: list[dict] = []

    for _, elem in ET.iterparse(str(xml_path), events=("end",)):
        if local_name(elem.tag) != "health-topic":
            continue

        language = normalize_space(elem.attrib.get("language", ""))
        if language_filter and language != language_filter:
            elem.clear()
            continue

        title = normalize_space(elem.attrib.get("title", ""))
        topic_id = normalize_space(elem.attrib.get("id", ""))
        url = normalize_space(elem.attrib.get("url", ""))
        date_created = normalize_space(elem.attrib.get("date-created", ""))
        date_modified = normalize_space(elem.attrib.get("date-modified", ""))

        also_called = dedupe_keep_order(text_content(x) for x in child_elements(elem, "also-called"))
        see_references = dedupe_keep_order(text_content(x) for x in child_elements(elem, "see-reference"))

        groups = []
        for x in child_elements(elem, "group"):
            groups.append(
                {
                    "id": normalize_space(x.attrib.get("id", "")),
                    "label": text_content(x),
                    "url": normalize_space(x.attrib.get("url", "")),
                }
            )

        related_topics = []
        for x in child_elements(elem, "related-topic"):
            related_topics.append(
                {
                    "id": normalize_space(x.attrib.get("id", "")),
                    "label": text_content(x),
                    "url": normalize_space(x.attrib.get("url", "")),
                }
            )

        mesh_headings = []
        for mh in child_elements(elem, "mesh-heading"):
            descriptor = first_child(mh, "descriptor")
            if descriptor is None:
                continue
            mesh_headings.append(
                {
                    "id": normalize_space(descriptor.attrib.get("id", "")),
                    "label": text_content(descriptor),
                }
            )

        full_summary = summary_html_to_text(inner_xml(first_child(elem, "full-summary")))
        alias_terms = dedupe_keep_order([title, *also_called, *see_references])

        row = {
            "topic_id": topic_id,
            "title": title,
            "url": url,
            "language": language,
            "date_created": date_created,
            "date_modified": date_modified,
            "also_called": also_called,
            "see_references": see_references,
            "groups": groups,
            "related_topics": related_topics,
            "mesh_headings": mesh_headings,
            "summary_text": full_summary,
            "alias_terms": alias_terms,
        }
        rows.append(row)
        elem.clear()

    rows.sort(key=lambda x: ((x["title"] or "").casefold(), x["topic_id"]))
    return rows


# ---------- MeSH Descriptor XML ----------

def _first_text(elem: ET.Element | None, xpath: str) -> str:
    if elem is None:
        return ""
    node = elem.find(xpath)
    return normalize_space(node.text if node is not None and node.text else "")


def _all_texts(elem: ET.Element | None, xpath: str) -> list[str]:
    if elem is None:
        return []
    values = []
    for node in elem.findall(xpath):
        text = normalize_space(node.text if node.text else "")
        if text:
            values.append(text)
    return dedupe_keep_order(values)


def parse_mesh_desc(xml_path: Path) -> list[dict]:
    rows: list[dict] = []

    for _, elem in ET.iterparse(str(xml_path), events=("end",)):
        if elem.tag != "DescriptorRecord":
            continue

        mesh_id = _first_text(elem, "DescriptorUI")
        label = _first_text(elem, "DescriptorName/String")
        if not mesh_id or not label:
            elem.clear()
            continue

        aliases = _all_texts(elem, ".//ConceptList/Concept/TermList/Term/String")
        aliases = [x for x in aliases if x.casefold() != label.casefold()]

        tree_numbers = _all_texts(elem, "TreeNumberList/TreeNumber")
        scope_note = _first_text(elem, ".//ConceptList/Concept/ScopeNote")

        rows.append(
            {
                "id": mesh_id,
                "label": label,
                "aliases": aliases,
                "tree_numbers": tree_numbers,
                "scope_note": scope_note,
            }
        )
        elem.clear()

    rows.sort(key=lambda x: ((x["label"] or "").casefold(), x["id"]))
    return rows


# ---------- Merge ----------

def merge_mesh_terms(mesh_rows: list[dict], topic_rows: list[dict]) -> list[dict]:
    store: dict[str, dict] = {}

    def ensure_entry(mesh_id: str, label: str) -> dict:
        key = mesh_id or label.casefold()
        entry = store.setdefault(
            key,
            {
                "id": mesh_id,
                "label": label,
                "aliases": [],
                "tree_numbers": [],
                "scope_note": "",
                "source_topic_ids": [],
                "source_topic_titles": [],
                "sources": [],
            },
        )
        if mesh_id and not entry["id"]:
            entry["id"] = mesh_id
        if label and not entry["label"]:
            entry["label"] = label
        return entry

    for row in mesh_rows:
        entry = ensure_entry(row["id"], row["label"])
        entry["aliases"] = dedupe_keep_order(entry["aliases"] + (row.get("aliases", []) or []))
        entry["tree_numbers"] = dedupe_keep_order(entry["tree_numbers"] + (row.get("tree_numbers", []) or []))
        if row.get("scope_note") and not entry["scope_note"]:
            entry["scope_note"] = row["scope_note"]
        if "mesh_desc_2026" not in entry["sources"]:
            entry["sources"].append("mesh_desc_2026")

    for topic in topic_rows:
        crosswalk_terms = dedupe_keep_order(
            [topic["title"], *topic["also_called"], *topic["see_references"]]
        )
        for mh in topic["mesh_headings"]:
            entry = ensure_entry(mh["id"], mh["label"])
            entry["aliases"] = dedupe_keep_order(entry["aliases"] + crosswalk_terms)
            entry["source_topic_ids"] = dedupe_keep_order(entry["source_topic_ids"] + [topic["topic_id"]])
            entry["source_topic_titles"] = dedupe_keep_order(entry["source_topic_titles"] + [topic["title"]])
            if "medlineplus_crosswalk" not in entry["sources"]:
                entry["sources"].append("medlineplus_crosswalk")

    merged = list(store.values())
    merged.sort(key=lambda x: ((x["label"] or "").casefold(), x["id"] or ""))
    return merged


# ---------- Retrieval units ----------

def render_taxonomy_unit(topic: dict) -> str:
    lines = [
        f"# {topic['title']}",
        "",
        "Section: taxonomy",
        f"Topic ID: {topic['topic_id']}",
    ]

    if topic["alias_terms"]:
        lines.append("Aliases: " + "; ".join(topic["alias_terms"]))
    if topic["mesh_headings"]:
        lines.append(
            "MeSH: " + "; ".join(
                f"{x['label']} ({x['id']})" if x["id"] else x["label"]
                for x in topic["mesh_headings"]
            )
        )
    if topic["groups"]:
        lines.append("Groups: " + "; ".join(x["label"] for x in topic["groups"] if x["label"]))
    if topic["related_topics"]:
        lines.append(
            "Related Topics: " + "; ".join(x["label"] for x in topic["related_topics"] if x["label"])
        )
    if topic["see_references"]:
        lines.append("See References: " + "; ".join(topic["see_references"]))
    lines.append("")
    return "\n".join(lines)


def render_summary_unit(topic: dict, text: str, part_idx: int, total_parts: int) -> str:
    lines = [
        f"# {topic['title']}",
        "",
        "Section: summary",
        f"Topic ID: {topic['topic_id']}",
        f"Summary Part: {part_idx}/{total_parts}",
    ]

    if topic["alias_terms"]:
        lines.append("Aliases: " + "; ".join(topic["alias_terms"][:8]))
    if topic["mesh_headings"]:
        lines.append(
            "MeSH: " + "; ".join(
                f"{x['label']} ({x['id']})" if x["id"] else x["label"]
                for x in topic["mesh_headings"][:4]
            )
        )

    lines.extend(["", text.strip(), ""])
    return "\n".join(lines)


def rebuild_retrieval_units(topic_rows: list[dict], out_dir: Path, summary_max_chars: int = 900) -> int:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for topic in topic_rows:
        base = f"{topic['topic_id']}_{slugify(topic['title'])}"

        taxonomy_path = out_dir / f"{base}__taxonomy.md"
        taxonomy_path.write_text(render_taxonomy_unit(topic), encoding="utf-8")
        count += 1

        parts = split_blocks(topic["summary_text"], max_chars=summary_max_chars)
        if not parts:
            continue

        total = len(parts)
        for idx, part in enumerate(parts, start=1):
            p = out_dir / f"{base}__summary_{idx:02d}.md"
            p.write_text(render_summary_unit(topic, part, idx, total), encoding="utf-8")
            count += 1

    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild MedlinePlus + MeSH derived resources")
    parser.add_argument(
        "--topics-xml",
        type=Path,
        default=Path("artifacts/raw/mplus_topics_2026-04-14.xml"),
    )
    parser.add_argument(
        "--mesh-desc-xml",
        type=Path,
        default=Path("artifacts/raw/desc2026.xml"),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("resources/medlineplus"),
    )
    parser.add_argument(
        "--language",
        type=str,
        default="English",
        help='默认 English；传 "all" 表示不过滤',
    )
    parser.add_argument(
        "--summary-max-chars",
        type=int,
        default=900,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/medlineplus/build_manifest.json"),
    )
    args = parser.parse_args()

    if not args.topics_xml.exists():
        raise FileNotFoundError(f"topics xml not found: {args.topics_xml}")
    if not args.mesh_desc_xml.exists():
        raise FileNotFoundError(f"mesh descriptor xml not found: {args.mesh_desc_xml}")

    args.out_root.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    language_filter = None if args.language.lower() == "all" else args.language

    topic_rows = parse_medlineplus_topics(args.topics_xml, language_filter=language_filter)
    mesh_rows = parse_mesh_desc(args.mesh_desc_xml)
    merged_mesh_rows = merge_mesh_terms(mesh_rows, topic_rows)

    topic_cards_path = args.out_root / "topic_cards.jsonl"
    mesh_terms_path = args.out_root / "mesh_terms.jsonl"
    retrieval_dir = args.out_root / "retrieval_units"

    write_jsonl(topic_cards_path, topic_rows)
    write_jsonl(mesh_terms_path, merged_mesh_rows)
    retrieval_count = rebuild_retrieval_units(
        topic_rows=topic_rows,
        out_dir=retrieval_dir,
        summary_max_chars=args.summary_max_chars,
    )

    manifest = {
        "topics_xml": str(args.topics_xml),
        "mesh_desc_xml": str(args.mesh_desc_xml),
        "topic_cards_path": str(topic_cards_path),
        "mesh_terms_path": str(mesh_terms_path),
        "retrieval_dir": str(retrieval_dir),
        "topic_count": len(topic_rows),
        "mesh_term_count": len(merged_mesh_rows),
        "retrieval_unit_count": retrieval_count,
        "language_filter": args.language,
        "summary_max_chars": args.summary_max_chars,
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
