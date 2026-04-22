#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import logging
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LOGGER = logging.getLogger("rebuild_medical_resources")

DEFAULT_TOPICS_XML = Path("artifacts/raw/mplus_topics_2026-04-14.xml")
DEFAULT_MESH_DESC_XML = Path("artifacts/raw/desc2026.xml")
DEFAULT_OUT_ROOT = Path("resources/medlineplus")
DEFAULT_MANIFEST = Path("artifacts/medlineplus/build_manifest.json")
DEFAULT_CHROMA_DIR = Path("chroma_db")
DEFAULT_COLLECTION_NAME = "medical_assistant"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text:latest"


# ---------------------------------------------------------------------------
# Logging / CLI
# ---------------------------------------------------------------------------


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s | %(message)s")


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def local_name(tag: str) -> str:
    if not tag:
        return ""
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


_WHITESPACE_RE = re.compile(r"\s+")


def normalize_space(text: str | None) -> str:
    if not text:
        return ""
    return _WHITESPACE_RE.sub(" ", text).strip()


_MULTILINE_WS_RE = re.compile(r"[ \t]+")
_MULTI_BREAK_RE = re.compile(r"\n{3,}")


def normalize_text_block(text: str | None) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [_MULTILINE_WS_RE.sub(" ", line).strip() for line in text.split("\n")]

    normalized_lines: list[str] = []
    previous_blank = True
    for line in lines:
        if line:
            normalized_lines.append(line)
            previous_blank = False
        elif not previous_blank:
            normalized_lines.append("")
            previous_blank = True

    joined = "\n".join(normalized_lines).strip()
    joined = _MULTI_BREAK_RE.sub("\n\n", joined)
    return joined


_SLUG_RE = re.compile(r"[^a-zA-Z0-9]+")


def slugify(text: str, *, default: str = "topic", max_len: int = 80) -> str:
    text = _SLUG_RE.sub("-", text).strip("-").lower()
    if not text:
        return default
    return text[:max_len] or default


T = Any


def dedupe_keep_order(values: Iterable[T]) -> list[T]:
    output: list[T] = []
    seen: set[str] = set()
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            key = value.casefold().strip()
            if not key:
                continue
        else:
            key = json.dumps(value, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        output.append(value)
    return output


_TAG_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"<\s*br\s*/?>", flags=re.I), "\n"),
    (re.compile(r"</\s*p\s*>", flags=re.I), "\n\n"),
    (re.compile(r"<\s*p[^>]*>", flags=re.I), ""),
    (re.compile(r"<\s*li[^>]*>", flags=re.I), "\n- "),
    (re.compile(r"</\s*(ul|ol)\s*>", flags=re.I), "\n"),
    (re.compile(r"<\s*(ul|ol)[^>]*>", flags=re.I), "\n"),
    (re.compile(r"</?\s*a[^>]*>", flags=re.I), ""),
)
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def html_fragment_to_text(fragment: str) -> str:
    text = fragment or ""
    for pattern, replacement in _TAG_REPLACEMENTS:
        text = pattern.sub(replacement, text)
    text = _HTML_TAG_RE.sub("", text)
    text = html.unescape(text)
    return normalize_text_block(text)


def element_text(elem: ET.Element | None) -> str:
    if elem is None:
        return ""
    return normalize_space(" ".join(part for part in elem.itertext()))


def element_inner_html(elem: ET.Element | None) -> str:
    if elem is None:
        return ""
    parts: list[str] = []
    if elem.text:
        parts.append(elem.text)
    for child in list(elem):
        parts.append(ET.tostring(child, encoding="unicode", method="xml"))
    return "".join(parts)


def child_elements(elem: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in list(elem) if local_name(child.tag) == name]


def first_child(elem: ET.Element, name: str) -> ET.Element | None:
    for child in list(elem):
        if local_name(child.tag) == name:
            return child
    return None


def first_child_text(elem: ET.Element, name: str) -> str:
    return element_text(first_child(elem, name))


def split_blocks(text: str, *, max_chars: int) -> list[str]:
    text = normalize_text_block(text)
    if not text:
        return []

    blocks = [block.strip() for block in text.split("\n\n") if block.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if not current:
            return
        chunks.append("\n\n".join(current).strip())
        current = []
        current_len = 0

    for block in blocks:
        if len(block) <= max_chars:
            projected = current_len + len(block) + (2 if current else 0)
            if current and projected > max_chars:
                flush()
            current.append(block)
            current_len += len(block) + (2 if len(current) > 1 else 0)
            continue

        lines = [line.strip() for line in block.split("\n") if line.strip()]
        for line in lines:
            if len(line) <= max_chars:
                projected = current_len + len(line) + (1 if current else 0)
                if current and projected > max_chars:
                    flush()
                current.append(line)
                current_len += len(line) + (1 if len(current) > 1 else 0)
                continue

            flush()
            for start in range(0, len(line), max_chars):
                piece = line[start : start + max_chars].strip()
                if piece:
                    chunks.append(piece)
        flush()

    flush()
    return [chunk for chunk in chunks if chunk]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def reset_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class LinkRef:
    label: str = ""
    id: str = ""
    url: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "label": self.label,
            "id": self.id,
            "url": self.url,
        }


@dataclass(slots=True)
class TopicRecord:
    topic_id: str
    title: str
    url: str = ""
    language: str = ""
    date_created: str = ""
    date_modified: str = ""
    meta_desc: str = ""
    summary_text: str = ""
    also_called: list[str] = field(default_factory=list)
    see_references: list[str] = field(default_factory=list)
    groups: list[LinkRef] = field(default_factory=list)
    related_topics: list[LinkRef] = field(default_factory=list)
    mesh_headings: list[LinkRef] = field(default_factory=list)
    alias_terms: list[str] = field(default_factory=list)

    def as_card_dict(self) -> dict[str, Any]:
        return {
            "topic_id": self.topic_id,
            "title": self.title,
            "url": self.url,
            "language": self.language,
            "date_created": self.date_created,
            "date_modified": self.date_modified,
            "meta_desc": self.meta_desc,
            "summary_text": self.summary_text,
            "also_called": self.also_called,
            "see_references": self.see_references,
            "groups": [ref.as_dict() for ref in self.groups],
            "related_topics": [ref.as_dict() for ref in self.related_topics],
            "mesh_headings": [ref.as_dict() for ref in self.mesh_headings],
            "alias_terms": self.alias_terms,
        }


@dataclass(slots=True)
class MeshDescriptor:
    id: str
    label: str
    aliases: list[str] = field(default_factory=list)
    tree_numbers: list[str] = field(default_factory=list)
    scope_note: str = ""
    linked_topic_ids: list[str] = field(default_factory=list)
    linked_topic_titles: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RetrievalChunk:
    chunk_id: str
    topic_id: str
    title: str
    section: str
    order: int
    text: str
    source_fields: list[str] = field(default_factory=list)
    alias_terms: list[str] = field(default_factory=list)
    mesh_ids: list[str] = field(default_factory=list)
    mesh_labels: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "topic_id": self.topic_id,
            "title": self.title,
            "section": self.section,
            "order": self.order,
            "text": self.text,
            "source_fields": self.source_fields,
            "alias_terms": self.alias_terms,
            "mesh_ids": self.mesh_ids,
            "mesh_labels": self.mesh_labels,
        }

    def as_markdown(self) -> str:
        lines = [
            f"# {self.title}",
            "",
            f"Section: {self.section}",
            f"Topic ID: {self.topic_id}",
            f"Chunk ID: {self.chunk_id}",
        ]
        if self.alias_terms:
            lines.append("Aliases: " + "; ".join(self.alias_terms))
        if self.mesh_labels:
            lines.append("MeSH: " + "; ".join(self.mesh_labels))
        lines.extend(["", self.text.strip(), ""])
        return "\n".join(lines)


@dataclass(slots=True)
class ChunkBuildConfig:
    summary_max_chars: int = 900
    include_overview: bool = True
    include_summary: bool = True
    include_relations: bool = True


# ---------------------------------------------------------------------------
# MedlinePlus topic parsing
# ---------------------------------------------------------------------------


def _parse_link_ref(elem: ET.Element) -> LinkRef:
    return LinkRef(
        label=element_text(elem),
        id=normalize_space(elem.attrib.get("id", "")),
        url=normalize_space(elem.attrib.get("url", "")),
    )


def _topic_language(elem: ET.Element) -> str:
    # The MedlinePlus export is usually English-only, but we keep this generic.
    for key, value in elem.attrib.items():
        if local_name(key) in {"language", "lang"}:
            return normalize_space(value)
    direct = first_child_text(elem, "language")
    if direct:
        return direct
    return ""


def parse_medlineplus_topics(xml_path: Path, *, language_filter: str | None = None) -> list[TopicRecord]:
    LOGGER.info("Parsing MedlinePlus topics from %s", xml_path)
    topics: list[TopicRecord] = []

    for _event, elem in ET.iterparse(xml_path, events=("end",)):
        if local_name(elem.tag) != "health-topic":
            continue

        language = _topic_language(elem)
        if language_filter and language and language.casefold() != language_filter.casefold():
            elem.clear()
            continue

        topic_id = normalize_space(elem.attrib.get("id", "")) or normalize_space(first_child_text(elem, "id"))
        title = first_child_text(elem, "title")
        url = normalize_space(elem.attrib.get("url", ""))
        date_created = normalize_space(elem.attrib.get("date-created", ""))
        date_modified = normalize_space(elem.attrib.get("date-modified", ""))
        meta_desc = first_child_text(elem, "meta-desc")

        full_summary_elem = first_child(elem, "full-summary")
        summary_text = html_fragment_to_text(element_inner_html(full_summary_elem))
        if not summary_text:
            summary_text = normalize_text_block(element_text(full_summary_elem))

        also_called = dedupe_keep_order(element_text(child) for child in child_elements(elem, "also-called"))
        see_references = dedupe_keep_order(element_text(child) for child in child_elements(elem, "see-reference"))
        groups = [_parse_link_ref(child) for child in child_elements(elem, "group") if element_text(child)]
        related_topics = [_parse_link_ref(child) for child in child_elements(elem, "related-topic") if element_text(child)]
        mesh_headings = [_parse_link_ref(child) for child in child_elements(elem, "mesh-heading") if element_text(child)]

        alias_terms = dedupe_keep_order([title, *also_called, *see_references])
        topic = TopicRecord(
            topic_id=topic_id or slugify(title, default="topic"),
            title=title,
            url=url,
            language=language,
            date_created=date_created,
            date_modified=date_modified,
            meta_desc=meta_desc,
            summary_text=summary_text,
            also_called=also_called,
            see_references=see_references,
            groups=groups,
            related_topics=related_topics,
            mesh_headings=mesh_headings,
            alias_terms=alias_terms,
        )
        topics.append(topic)
        elem.clear()

    topics.sort(key=lambda item: (item.title.casefold(), item.topic_id))
    LOGGER.info("Parsed %d topic rows", len(topics))
    return topics


# ---------------------------------------------------------------------------
# MeSH descriptor parsing (reference only)
# ---------------------------------------------------------------------------


def _find_first_text_by_path(elem: ET.Element, parts: Sequence[str]) -> str:
    current: ET.Element | None = elem
    for part in parts:
        if current is None:
            return ""
        next_node = None
        for child in list(current):
            if local_name(child.tag) == part:
                next_node = child
                break
        current = next_node
    return element_text(current)


def _find_all_texts_by_path(elem: ET.Element, parts: Sequence[str]) -> list[str]:
    nodes = [elem]
    for part in parts:
        next_nodes: list[ET.Element] = []
        for node in nodes:
            for child in list(node):
                if local_name(child.tag) == part:
                    next_nodes.append(child)
        nodes = next_nodes
        if not nodes:
            return []
    return [text for text in (element_text(node) for node in nodes) if text]


def parse_mesh_descriptors(xml_path: Path) -> list[MeshDescriptor]:
    LOGGER.info("Parsing MeSH descriptors from %s", xml_path)
    rows: list[MeshDescriptor] = []

    for _event, elem in ET.iterparse(xml_path, events=("end",)):
        if local_name(elem.tag) != "DescriptorRecord":
            continue

        mesh_id = _find_first_text_by_path(elem, ["DescriptorUI"])
        label = _find_first_text_by_path(elem, ["DescriptorName", "String"])
        aliases = _find_all_texts_by_path(elem, ["ConceptList", "Concept", "TermList", "Term", "String"])
        aliases = [alias for alias in dedupe_keep_order(aliases) if alias.casefold() != label.casefold()]
        tree_numbers = dedupe_keep_order(_find_all_texts_by_path(elem, ["TreeNumberList", "TreeNumber"]))
        scope_note = _find_first_text_by_path(elem, ["ConceptList", "Concept", "ScopeNote"])

        if mesh_id and label:
            rows.append(
                MeshDescriptor(
                    id=mesh_id,
                    label=label,
                    aliases=aliases,
                    tree_numbers=tree_numbers,
                    scope_note=scope_note,
                )
            )
        elem.clear()

    rows.sort(key=lambda item: (item.label.casefold(), item.id))
    LOGGER.info("Parsed %d mesh descriptors", len(rows))
    return rows


def build_mesh_reference(topics: Sequence[TopicRecord], mesh_rows: Sequence[MeshDescriptor]) -> list[MeshDescriptor]:
    by_id = {row.id: row for row in mesh_rows if row.id}
    by_label = {row.label.casefold(): row for row in mesh_rows if row.label}

    topic_links: dict[str, dict[str, set[str]]] = {}
    for topic in topics:
        for heading in topic.mesh_headings:
            target: MeshDescriptor | None = None
            if heading.id and heading.id in by_id:
                target = by_id[heading.id]
            elif heading.label and heading.label.casefold() in by_label:
                target = by_label[heading.label.casefold()]
            if target is None:
                continue
            bucket = topic_links.setdefault(target.id, {"ids": set(), "titles": set()})
            bucket["ids"].add(topic.topic_id)
            bucket["titles"].add(topic.title)

    references: list[MeshDescriptor] = []
    for mesh_id, link_info in topic_links.items():
        row = by_id[mesh_id]
        references.append(
            MeshDescriptor(
                id=row.id,
                label=row.label,
                aliases=row.aliases,
                tree_numbers=row.tree_numbers,
                scope_note=row.scope_note,
                linked_topic_ids=sorted(link_info["ids"]),
                linked_topic_titles=sorted(link_info["titles"]),
            )
        )

    references.sort(key=lambda item: (item.label.casefold(), item.id))
    return references


# ---------------------------------------------------------------------------
# Rendering + chunk generation
# ---------------------------------------------------------------------------


def render_topic_markdown(topic: TopicRecord) -> str:
    lines = [f"# {topic.title}", ""]
    lines.append(f"Topic ID: {topic.topic_id}")
    if topic.url:
        lines.append(f"Source URL: {topic.url}")
    if topic.language:
        lines.append(f"Language: {topic.language}")
    if topic.date_modified:
        lines.append(f"Last Updated: {topic.date_modified}")
    if topic.alias_terms:
        lines.append("Aliases: " + "; ".join(topic.alias_terms))
    if topic.mesh_headings:
        mesh_text = "; ".join(
            f"{ref.label} ({ref.id})" if ref.id else ref.label for ref in topic.mesh_headings if ref.label
        )
        if mesh_text:
            lines.append("MeSH Headings: " + mesh_text)

    if topic.meta_desc:
        lines.extend(["", "## Overview", topic.meta_desc])
    if topic.summary_text:
        lines.extend(["", "## Summary", topic.summary_text])
    if topic.see_references:
        lines.extend(["", "## See References", "- " + "\n- ".join(topic.see_references)])
    if topic.groups:
        lines.extend(["", "## Groups", "- " + "\n- ".join(ref.label for ref in topic.groups if ref.label)])
    if topic.related_topics:
        lines.extend(["", "## Related Topics", "- " + "\n- ".join(ref.label for ref in topic.related_topics if ref.label)])
    lines.append("")
    return "\n".join(lines)


def _build_overview_text(topic: TopicRecord) -> str:
    lines: list[str] = []
    if topic.alias_terms:
        lines.append("Aliases: " + "; ".join(topic.alias_terms))
    if topic.meta_desc:
        lines.append(topic.meta_desc)
    if topic.see_references:
        lines.append("See references: " + "; ".join(topic.see_references))
    return "\n\n".join(lines).strip()


def _build_relations_text(topic: TopicRecord) -> str:
    lines: list[str] = []
    if topic.groups:
        lines.append("Groups: " + "; ".join(ref.label for ref in topic.groups if ref.label))
    if topic.related_topics:
        lines.append("Related topics: " + "; ".join(ref.label for ref in topic.related_topics if ref.label))
    if topic.mesh_headings:
        lines.append(
            "Reference headings: "
            + "; ".join(
                f"{ref.label} ({ref.id})" if ref.id else ref.label for ref in topic.mesh_headings if ref.label
            )
        )
    return "\n\n".join(lines).strip()


def build_retrieval_chunks(
    topics: Sequence[TopicRecord],
    *,
    config: ChunkBuildConfig,
) -> list[RetrievalChunk]:
    chunks: list[RetrievalChunk] = []

    for topic in topics:
        mesh_ids = [ref.id for ref in topic.mesh_headings if ref.id]
        mesh_labels = [ref.label for ref in topic.mesh_headings if ref.label]
        topic_slug = slugify(topic.topic_id or topic.title, default="topic")

        if config.include_overview:
            overview_text = _build_overview_text(topic)
            if overview_text:
                chunks.append(
                    RetrievalChunk(
                        chunk_id=f"{topic_slug}:overview:01",
                        topic_id=topic.topic_id,
                        title=topic.title,
                        section="overview",
                        order=1,
                        text=overview_text,
                        source_fields=["title", "also_called", "see_reference", "meta_desc"],
                        alias_terms=topic.alias_terms,
                        mesh_ids=mesh_ids,
                        mesh_labels=mesh_labels,
                    )
                )

        if config.include_summary:
            summary_source = "\n\n".join(part for part in [topic.meta_desc, topic.summary_text] if part).strip()
            parts = split_blocks(summary_source, max_chars=config.summary_max_chars)
            for idx, part in enumerate(parts, start=1):
                chunks.append(
                    RetrievalChunk(
                        chunk_id=f"{topic_slug}:summary:{idx:02d}",
                        topic_id=topic.topic_id,
                        title=topic.title,
                        section="summary",
                        order=idx,
                        text=part,
                        source_fields=["meta_desc", "summary_text"],
                        alias_terms=topic.alias_terms,
                        mesh_ids=mesh_ids,
                        mesh_labels=mesh_labels,
                    )
                )

        if config.include_relations:
            relations_text = _build_relations_text(topic)
            if relations_text:
                chunks.append(
                    RetrievalChunk(
                        chunk_id=f"{topic_slug}:relations:01",
                        topic_id=topic.topic_id,
                        title=topic.title,
                        section="relations",
                        order=1,
                        text=relations_text,
                        source_fields=["group", "related_topic", "mesh_heading"],
                        alias_terms=topic.alias_terms,
                        mesh_ids=mesh_ids,
                        mesh_labels=mesh_labels,
                    )
                )

    chunks.sort(key=lambda item: (item.title.casefold(), item.section, item.order, item.chunk_id))
    return chunks


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def write_topic_docs(topics: Sequence[TopicRecord], out_dir: Path) -> int:
    reset_directory(out_dir)
    count = 0
    for topic in topics:
        base = slugify(topic.topic_id or topic.title, default="topic")
        path = out_dir / f"{base}.md"
        path.write_text(render_topic_markdown(topic), encoding="utf-8")
        count += 1
    return count


def write_retrieval_units(chunks: Sequence[RetrievalChunk], out_dir: Path) -> int:
    reset_directory(out_dir)
    count = 0
    for chunk in chunks:
        safe_name = chunk.chunk_id.replace(":", "__")
        path = out_dir / f"{safe_name}.md"
        path.write_text(chunk.as_markdown(), encoding="utf-8")
        count += 1
    return count


# ---------------------------------------------------------------------------
# Optional Chroma build
# ---------------------------------------------------------------------------


def _primitive_metadata(chunk: RetrievalChunk) -> dict[str, Any]:
    return {
        "source": f"topic://{chunk.topic_id}/{chunk.section}/{chunk.order}",
        "title": chunk.title,
        "chunk_id": chunk.chunk_id,
        "topic_id": chunk.topic_id,
        "section": chunk.section,
        "order": chunk.order,
        "alias_terms": " | ".join(chunk.alias_terms),
        "mesh_ids": " | ".join(chunk.mesh_ids),
        "mesh_labels": " | ".join(chunk.mesh_labels),
        "source_fields": " | ".join(chunk.source_fields),
    }


def build_chroma_index(
    chunks: Sequence[RetrievalChunk],
    *,
    chroma_dir: Path,
    collection_name: str,
    embedding_model: str,
    batch_size: int,
    recreate: bool,
) -> dict[str, Any]:
    try:
        from langchain_chroma import Chroma
        from langchain_core.documents import Document
        from langchain_ollama import OllamaEmbeddings
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError(
            "Index building requires langchain_chroma, langchain_core, and langchain_ollama."
        ) from exc

    chroma_dir.mkdir(parents=True, exist_ok=True)
    embeddings = OllamaEmbeddings(model=embedding_model)
    vectorstore = Chroma(
        collection_name=collection_name,
        persist_directory=str(chroma_dir),
        embedding_function=embeddings,
    )

    if recreate:
        try:
            vectorstore.delete_collection()
        except Exception:
            LOGGER.debug("Chroma collection deletion skipped", exc_info=True)
        vectorstore = Chroma(
            collection_name=collection_name,
            persist_directory=str(chroma_dir),
            embedding_function=embeddings,
        )

    documents = [Document(page_content=chunk.as_markdown(), metadata=_primitive_metadata(chunk)) for chunk in chunks]
    effective_batch_size = batch_size
    try:
        max_batch_size = int(vectorstore._client.get_max_batch_size())  # type: ignore[attr-defined]
        effective_batch_size = min(batch_size, max_batch_size)
    except Exception:
        max_batch_size = None

    total = len(documents)
    done = 0
    for start in range(0, total, effective_batch_size):
        batch = documents[start : start + effective_batch_size]
        vectorstore.add_documents(batch)
        done += len(batch)
        LOGGER.info("Indexed %d/%d chunks", done, total)

    return {
        "chroma_dir": str(chroma_dir),
        "collection_name": collection_name,
        "embedding_model": embedding_model,
        "indexed_documents": total,
        "max_batch_size": max_batch_size,
        "effective_batch_size": effective_batch_size,
    }


# ---------------------------------------------------------------------------
# Default inference helpers
# ---------------------------------------------------------------------------


def infer_repo_defaults() -> dict[str, Any]:
    defaults = {
        "chroma_dir": DEFAULT_CHROMA_DIR,
        "collection_name": DEFAULT_COLLECTION_NAME,
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
    }
    try:
        from medical_assistant.config import get_settings

        settings = get_settings()
        defaults["chroma_dir"] = Path(settings.chroma_dir)
        defaults["collection_name"] = settings.collection_name
        defaults["embedding_model"] = settings.embedding_model
    except Exception:
        LOGGER.debug("Could not infer defaults from medical_assistant.config", exc_info=True)
    return defaults


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    inferred = infer_repo_defaults()
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild MedlinePlus-derived retrieval resources. "
            "This script treats MedlinePlus topic content as the primary retrieval source "
            "and keeps MeSH as optional reference data."
        )
    )
    parser.add_argument("--topics-xml", type=Path, default=DEFAULT_TOPICS_XML)
    parser.add_argument(
        "--mesh-desc-xml",
        type=Path,
        default=DEFAULT_MESH_DESC_XML,
        help="Optional MeSH descriptor XML. Missing files are skipped unless --require-mesh is set.",
    )
    parser.add_argument("--require-mesh", action="store_true", help="Fail if --mesh-desc-xml does not exist.")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--language",
        type=str,
        default="English",
        help='Language filter. Use "all" to skip language filtering.',
    )
    parser.add_argument("--summary-max-chars", type=int, default=900)
    parser.add_argument("--no-topic-docs", action="store_true", help="Skip writing full topic markdown files.")
    parser.add_argument(
        "--no-retrieval-units",
        action="store_true",
        help="Skip writing retrieval-unit markdown files.",
    )
    parser.add_argument(
        "--build-index",
        action="store_true",
        help="Build the Chroma index from generated retrieval chunks.",
    )
    parser.add_argument("--chroma-dir", type=Path, default=inferred["chroma_dir"])
    parser.add_argument("--collection-name", type=str, default=inferred["collection_name"])
    parser.add_argument("--embedding-model", type=str, default=inferred["embedding_model"])
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument(
        "--keep-existing-output",
        action="store_true",
        help="Do not clear topic_docs/ or retrieval_units/ before writing.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def maybe_prepare_dir(path: Path, *, clean: bool) -> None:
    if clean:
        reset_directory(path)
    else:
        path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    if not args.topics_xml.exists():
        raise FileNotFoundError(f"topics xml not found: {args.topics_xml}")

    if args.require_mesh and not args.mesh_desc_xml.exists():
        raise FileNotFoundError(f"mesh descriptor xml not found: {args.mesh_desc_xml}")

    args.out_root.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    language_filter = None if args.language.lower() == "all" else args.language
    topic_rows = parse_medlineplus_topics(args.topics_xml, language_filter=language_filter)

    mesh_rows: list[MeshDescriptor] = []
    mesh_reference_rows: list[MeshDescriptor] = []
    if args.mesh_desc_xml.exists():
        mesh_rows = parse_mesh_descriptors(args.mesh_desc_xml)
        mesh_reference_rows = build_mesh_reference(topic_rows, mesh_rows)
    else:
        LOGGER.info("MeSH descriptor XML not found, skipping MeSH reference build: %s", args.mesh_desc_xml)

    chunk_config = ChunkBuildConfig(summary_max_chars=max(200, int(args.summary_max_chars)))
    retrieval_chunks = build_retrieval_chunks(topic_rows, config=chunk_config)

    topic_cards_path = args.out_root / "topic_cards.jsonl"
    mesh_terms_path = args.out_root / "mesh_terms.jsonl"
    retrieval_chunks_path = args.out_root / "retrieval_chunks.jsonl"
    topic_docs_dir = args.out_root / "topic_docs"
    retrieval_units_dir = args.out_root / "retrieval_units"

    write_jsonl(topic_cards_path, (topic.as_card_dict() for topic in topic_rows))
    write_jsonl(mesh_terms_path, (row.as_dict() for row in mesh_reference_rows))
    write_jsonl(retrieval_chunks_path, (chunk.as_dict() for chunk in retrieval_chunks))

    topic_doc_count = 0
    retrieval_unit_count = 0

    if not args.no_topic_docs:
        if args.keep_existing_output:
            topic_docs_dir.mkdir(parents=True, exist_ok=True)
        topic_doc_count = write_topic_docs(topic_rows, topic_docs_dir)

    if not args.no_retrieval_units:
        if args.keep_existing_output:
            retrieval_units_dir.mkdir(parents=True, exist_ok=True)
        retrieval_unit_count = write_retrieval_units(retrieval_chunks, retrieval_units_dir)

    index_info: dict[str, Any] | None = None
    if args.build_index:
        index_info = build_chroma_index(
            retrieval_chunks,
            chroma_dir=args.chroma_dir,
            collection_name=args.collection_name,
            embedding_model=args.embedding_model,
            batch_size=max(1, int(args.batch_size)),
            recreate=True,
        )

    manifest = {
        "topics_xml": str(args.topics_xml),
        "mesh_desc_xml": str(args.mesh_desc_xml) if args.mesh_desc_xml.exists() else None,
        "out_root": str(args.out_root),
        "topic_cards_path": str(topic_cards_path),
        "mesh_terms_path": str(mesh_terms_path),
        "retrieval_chunks_path": str(retrieval_chunks_path),
        "topic_docs_dir": str(topic_docs_dir),
        "retrieval_units_dir": str(retrieval_units_dir),
        "topic_count": len(topic_rows),
        "mesh_descriptor_count": len(mesh_rows),
        "mesh_reference_count": len(mesh_reference_rows),
        "retrieval_chunk_count": len(retrieval_chunks),
        "topic_doc_count": topic_doc_count,
        "retrieval_unit_count": retrieval_unit_count,
        "language_filter": args.language,
        "summary_max_chars": chunk_config.summary_max_chars,
        "build_index": bool(args.build_index),
        "index": index_info,
    }

    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
