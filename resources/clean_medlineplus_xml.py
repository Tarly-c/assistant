from __future__ import annotations

import argparse
import html
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

ALLOWED_SITE_CATEGORIES = {
    "Start Here",
    "Learn More",
    "Patient Handouts",
    "Encyclopedia",
    "Diagnosis and Tests",
    "Treatments and Therapies",
    "Specifics",
    "Children",
    "Teenagers",
    "Women",
    "Men",
    "Seniors",
}


def clean_html_fragment(text: str) -> str:
    text = html.unescape(text or "")
    text = re.sub(r"<\s*br\s*/?\s*>", "\n", text, flags=re.I)
    text = re.sub(r"<\s*/\s*p\s*>", "\n\n", text, flags=re.I)
    text = re.sub(r"<\s*p[^>]*>", "", text, flags=re.I)
    text = re.sub(r"<\s*/\s*li\s*>", "\n", text, flags=re.I)
    text = re.sub(r"<\s*li[^>]*>", "- ", text, flags=re.I)
    text = re.sub(r"<\s*/\s*ul\s*>", "\n", text, flags=re.I)
    text = re.sub(r"<\s*ul[^>]*>", "", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def text_of(node: ET.Element | None) -> str:
    if node is None:
        return ""
    return "".join(node.itertext()).strip()


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    return text or "untitled"


def unique_keep_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        value = (item or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def extract_topic(topic: ET.Element, keep_sites: int = 8) -> dict:
    title = topic.get("title", "").strip()
    url = topic.get("url", "").strip()
    topic_id = topic.get("id", "").strip()
    language = topic.get("language", "").strip()
    date_created = topic.get("date-created", "").strip()
    meta_desc = topic.get("meta-desc", "").strip()

    aliases = unique_keep_order(text_of(x) for x in topic.findall("also-called"))
    groups = unique_keep_order(text_of(x) for x in topic.findall("group"))
    related_topics = unique_keep_order(text_of(x) for x in topic.findall("related-topic"))
    see_refs = unique_keep_order(text_of(x) for x in topic.findall("see-reference"))
    mesh_terms = unique_keep_order(text_of(x) for x in topic.findall("./mesh-heading/descriptor"))
    other_languages = unique_keep_order(text_of(x) for x in topic.findall("other-language"))
    primary_institute = text_of(topic.find("primary-institute"))

    summary = clean_html_fragment(text_of(topic.find("full-summary")))

    sites = []
    for site in topic.findall("site"):
        site_title = site.get("title", "").strip()
        site_url = site.get("url", "").strip()
        categories = unique_keep_order(text_of(x) for x in site.findall("information-category"))
        orgs = unique_keep_order(text_of(x) for x in site.findall("organization"))
        if not site_title or not site_url:
            continue
        if categories and not any(c in ALLOWED_SITE_CATEGORIES for c in categories):
            continue
        sites.append(
            {
                "title": site_title,
                "url": site_url,
                "categories": categories,
                "organizations": orgs,
            }
        )
        if len(sites) >= keep_sites:
            break

    return {
        "id": topic_id,
        "title": title,
        "url": url,
        "language": language,
        "date_created": date_created,
        "meta_desc": meta_desc,
        "aliases": aliases,
        "groups": groups,
        "mesh_terms": mesh_terms,
        "related_topics": related_topics,
        "see_also": see_refs,
        "other_languages": other_languages,
        "primary_institute": primary_institute,
        "summary": summary,
        "sites": sites,
    }


def render_markdown(entry: dict) -> str:
    lines: list[str] = []
    lines.append(f"# {entry['title']}")
    lines.append("")
    lines.append(f"- Topic ID: {entry['id']}")
    lines.append(f"- Language: {entry['language']}")
    lines.append(f"- URL: {entry['url']}")
    if entry.get("date_created"):
        lines.append(f"- Date created: {entry['date_created']}")
    if entry.get("primary_institute"):
        lines.append(f"- Primary institute: {entry['primary_institute']}")
    lines.append("")

    if entry.get("aliases"):
        lines.append("## Also called")
        lines.extend(f"- {x}" for x in entry["aliases"])
        lines.append("")

    if entry.get("groups"):
        lines.append("## Groups")
        lines.extend(f"- {x}" for x in entry["groups"])
        lines.append("")

    if entry.get("mesh_terms"):
        lines.append("## MeSH")
        lines.extend(f"- {x}" for x in entry["mesh_terms"])
        lines.append("")

    if entry.get("related_topics"):
        lines.append("## Related topics")
        lines.extend(f"- {x}" for x in entry["related_topics"])
        lines.append("")

    if entry.get("see_also"):
        lines.append("## See also")
        lines.extend(f"- {x}" for x in entry["see_also"])
        lines.append("")

    if entry.get("meta_desc"):
        lines.append("## Meta description")
        lines.append(entry["meta_desc"])
        lines.append("")

    if entry.get("summary"):
        lines.append("## Summary")
        lines.append(entry["summary"])
        lines.append("")

    if entry.get("sites"):
        lines.append("## Selected references")
        for site in entry["sites"]:
            cats = ", ".join(site.get("categories") or [])
            orgs = ", ".join(site.get("organizations") or [])
            bullet = f"- {site['title']}"
            if cats:
                bullet += f" [{cats}]"
            if orgs:
                bullet += f" — {orgs}"
            bullet += f" — {site['url']}"
            lines.append(bullet)
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean MedlinePlus XML into per-topic markdown files.")
    parser.add_argument("--xml", required=True, help="Path to mplus_topics XML file")
    parser.add_argument(
        "--out",
        default="resources",
        help="Output directory. Use resources to work with the current local_paper_search.py",
    )
    parser.add_argument(
        "--language",
        default="English",
        help="Keep only one language. Use ALL to keep every language.",
    )
    parser.add_argument(
        "--prefix",
        default="medlineplus",
        help="Filename prefix for generated markdown files",
    )
    parser.add_argument(
        "--keep-sites",
        type=int,
        default=8,
        help="Max number of site links kept per topic",
    )
    parser.add_argument(
        "--jsonl",
        action="store_true",
        help="Also write a medlineplus_index.jsonl manifest",
    )
    args = parser.parse_args()

    xml_path = Path(args.xml)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    count = 0
    manifest_path = out_dir / f"{args.prefix}_index.jsonl"
    manifest_fp = manifest_path.open("w", encoding="utf-8") if args.jsonl else None

    try:
        for topic in root.findall("health-topic"):
            language = topic.get("language", "").strip()
            if args.language != "ALL" and language != args.language:
                continue

            entry = extract_topic(topic, keep_sites=args.keep_sites)
            if not entry.get("title") or not entry.get("summary"):
                continue

            filename = f"{args.prefix}_{slugify(entry['title'])}_{entry['id']}.md"
            file_path = out_dir / filename
            file_path.write_text(render_markdown(entry), encoding="utf-8")

            if manifest_fp is not None:
                manifest_fp.write(json.dumps({**entry, "file": filename}, ensure_ascii=False) + "\n")

            count += 1
    finally:
        if manifest_fp is not None:
            manifest_fp.close()

    print(f"Generated {count} markdown files in {out_dir}")
    if args.jsonl:
        print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
