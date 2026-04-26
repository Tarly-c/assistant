from __future__ import annotations

"""Dynamic, data-derived probe mining for the case-localization workflow.

This module deliberately avoids hand-written symptom/metadata/answer word lists.
It builds candidate probes from the cases themselves:

1. broad anchor probes from title/description/optional bilingual fields;
2. local observation-window probes from case text clustering.

The tree builder decides whether the best candidate is worth asking. This module
should return candidates whenever a split can be computed, even if the gain is
low, so build diagnostics can show why a node stopped.
"""

from dataclasses import dataclass, field
from hashlib import sha1
import math
import re
from typing import Any, Iterable, Sequence

from medical_assistant.schemas import CaseCandidate, CaseRecord, PlannedQuestion


@dataclass(frozen=True)
class TextUnit:
    case_id: str
    text: str
    source: str = "description"


@dataclass
class ProbeCandidate:
    probe_id: str
    label: str
    prototype_text: str
    question_seed: str
    positive_case_ids: list[str]
    negative_case_ids: list[str]
    unknown_case_ids: list[str] = field(default_factory=list)
    evidence_texts: list[str] = field(default_factory=list)
    split_score: float = 0.0
    debug: dict[str, Any] = field(default_factory=dict)

    def to_planned_question(
        self,
        *,
        question_id: str | None = None,
        tree_node_id: str = "",
        yes_child_id: str = "",
        no_child_id: str = "",
        strategy: str = "local_dynamic_probe",
    ) -> PlannedQuestion:
        return PlannedQuestion(
            question_id=question_id or f"q_{self.probe_id}",
            feature_id=self.probe_id,
            label=self.label,
            text=self.question_seed,
            positive_case_ids=self.positive_case_ids,
            negative_case_ids=self.negative_case_ids,
            unknown_case_ids=self.unknown_case_ids,
            split_score=round(float(self.split_score), 4),
            strategy=strategy,
            tree_node_id=tree_node_id,
            yes_child_id=yes_child_id,
            no_child_id=no_child_id,
            evidence_texts=self.evidence_texts[:5],
            debug=self.debug,
        )


# ---------------------------------------------------------------------------
# Compatibility shims for older imports. The new workflow has no global static
# feature table.


def all_features() -> tuple[object, ...]:
    return ()


def get_feature(feature_id: str) -> None:
    return None


def extract_features(text: str) -> list[str]:
    return []


def feature_labels(feature_ids: Iterable[str]) -> list[str]:
    return [str(fid) for fid in feature_ids if fid]


# ---------------------------------------------------------------------------
# Generic text processing.


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").strip().lower())


def readable_text(text: str) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    return text.strip(" ，,。.;；、")


def _is_observable_unit(text: str) -> bool:
    compact = _norm(readable_text(text))
    return 4 <= len(compact) <= 180


def _dedupe_keep_order(items: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        cleaned = readable_text(item)
        key = _norm(cleaned)
        if cleaned and key not in seen:
            out.append(cleaned)
            seen.add(key)
    return out


def _case_extra_texts(case: CaseRecord | CaseCandidate) -> list[str]:
    """Collect optional bilingual/search fields if the schema/data provides them.

    This makes the search space bilingual without requiring the code to contain
    a hand-written translation table. Existing Chinese-only JSON continues to
    work; if future case records include title_en, description_en, aliases,
    keywords, or key_terms_en, those texts participate in the same mining and
    retrieval flow.
    """

    texts: list[str] = []
    for attr in (
        "title_en",
        "description_en",
        "aliases",
        "keywords",
        "key_terms_en",
        "search_terms",
    ):
        value = getattr(case, attr, None)
        if isinstance(value, str):
            texts.append(value)
        elif isinstance(value, (list, tuple, set)):
            texts.extend(str(x) for x in value if x)
    return texts


def case_search_text(case: CaseRecord | CaseCandidate) -> str:
    parts = [case.title, case.description, case.treat]
    parts.extend(case.feature_tags or [])
    parts.extend(_case_extra_texts(case))
    return "\n".join(readable_text(str(p)) for p in parts if readable_text(str(p)))


def split_observation_units(
    title: str = "",
    description: str = "",
    *,
    include_title: bool = False,
    extra_texts: Sequence[str] | None = None,
) -> list[str]:
    """Split text into context-preserving windows.

    No dependent-prefix table is used. Instead, every adjacent 1/2/3-clause
    window is emitted, so context-dependent fragments can still be represented
    together with their neighbors.
    """

    raw_parts: list[str] = [description or ""]
    raw_parts.extend(extra_texts or [])
    raw = "\n".join(part for part in raw_parts if part)
    clauses = [readable_text(x) for x in re.split(r"[。；;！!？?\n]+", raw) if readable_text(x)]

    units: list[str] = []
    if include_title and title:
        units.append(readable_text(title))

    for i in range(len(clauses)):
        for width in (1, 2, 3):
            window = readable_text("，".join(clauses[i : i + width]))
            if window and _is_observable_unit(window):
                units.append(window)

    return _dedupe_keep_order(units)


def split_observable_units(case: CaseRecord | CaseCandidate) -> list[str]:
    return split_observation_units(
        case.title,
        case.description,
        include_title=False,
        extra_texts=_case_extra_texts(case),
    )


def collect_text_units(cases: Sequence[CaseRecord | CaseCandidate]) -> list[TextUnit]:
    units: list[TextUnit] = []
    for case in cases:
        for text in split_observable_units(case):
            units.append(TextUnit(case_id=case.case_id, text=text, source="description"))
    return units


# ---------------------------------------------------------------------------
# Lightweight bilingual lexical vectorization.


def _char_ngrams(text: str, ns: tuple[int, ...] = (2, 3, 4)) -> list[str]:
    compact = _norm(text)
    grams: list[str] = []
    for n in ns:
        if len(compact) >= n:
            grams.extend(compact[i : i + n] for i in range(len(compact) - n + 1))
    grams.extend(re.findall(r"[a-z][a-z0-9_/-]{1,}", compact))
    return grams


def _latin_terms(text: str) -> list[str]:
    words = re.findall(r"[a-z][a-z0-9_/-]{2,}", (text or "").lower())
    terms: list[str] = []
    for i, word in enumerate(words):
        terms.append(word)
        if i + 1 < len(words):
            terms.append(f"{word} {words[i + 1]}")
        if i + 2 < len(words):
            terms.append(f"{word} {words[i + 1]} {words[i + 2]}")
    return terms


def _cjk_terms(text: str, ns: tuple[int, ...] = (2, 3, 4)) -> list[str]:
    compact = _norm(re.sub(r"[^\u4e00-\u9fff]+", "", text or ""))
    terms: list[str] = []
    for n in ns:
        if len(compact) >= n:
            terms.extend(compact[i : i + n] for i in range(len(compact) - n + 1))
    return terms


def _candidate_terms(text: str) -> list[str]:
    return _dedupe_keep_order([*_cjk_terms(text), *_latin_terms(text)])


def build_idf(texts: Sequence[str]) -> dict[str, float]:
    doc_count: dict[str, int] = {}
    for text in texts:
        for gram in set(_char_ngrams(text)):
            doc_count[gram] = doc_count.get(gram, 0) + 1
    n_docs = max(1, len(texts))
    return {gram: math.log((1 + n_docs) / (1 + df)) + 1.0 for gram, df in doc_count.items()}


def vectorize(text: str, idf: dict[str, float] | None = None) -> dict[str, float]:
    idf = idf or {}
    counts: dict[str, float] = {}
    for gram in _char_ngrams(text):
        counts[gram] = counts.get(gram, 0.0) + 1.0
    weighted = {gram: count * idf.get(gram, 1.0) for gram, count in counts.items()}
    norm = math.sqrt(sum(v * v for v in weighted.values())) or 1.0
    return {gram: value / norm for gram, value in weighted.items()}


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    return sum(value * b.get(key, 0.0) for key, value in a.items())


# ---------------------------------------------------------------------------
# Split scoring.


def split_quality(pos: int, neg: int, unknown: int, total: int) -> float:
    if total <= 0 or pos <= 0 or neg <= 0:
        return 0.0
    coverage = (pos + neg) / total
    balance = 1.0 - abs(pos - neg) / max(1, pos + neg)
    unknown_penalty = 1.0 - (unknown / total)
    p = pos / max(1, pos + neg)
    entropy = 0.0
    if 0.0 < p < 1.0:
        entropy = -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
    return max(0.0, entropy * coverage * (balance ** 1.15) * unknown_penalty)


def _best_threshold_split(
    sims_by_case: dict[str, float],
    *,
    margin: float = 0.025,
    min_child_size: int = 1,
    min_child_ratio: float = 0.0,
) -> tuple[list[str], list[str], list[str], float, float]:
    total = len(sims_by_case)
    if total <= 1:
        return list(sims_by_case), [], [], 0.0, 1.0

    values = sorted(set(round(v, 4) for v in sims_by_case.values()), reverse=True)
    if len(values) >= 2:
        thresholds = [(values[i] + values[i + 1]) / 2 for i in range(len(values) - 1)]
    else:
        thresholds = [values[0] if values else 0.0]

    best: tuple[list[str], list[str], list[str], float, float] | None = None
    best_score = -1.0
    for threshold in thresholds:
        if threshold <= 0.0:
            continue
        positive = [cid for cid, sim in sims_by_case.items() if sim >= threshold + margin]
        unknown = [cid for cid, sim in sims_by_case.items() if threshold - margin <= sim < threshold + margin]
        negative = [cid for cid, sim in sims_by_case.items() if sim < threshold - margin]
        min_child = max(1, min_child_size, int(total * min_child_ratio))
        if len(positive) < min_child or len(negative) < min_child:
            continue
        score = split_quality(len(positive), len(negative), len(unknown), total)
        if score > best_score:
            best_score = score
            best = (positive, negative, unknown, score, threshold)

    if best is None:
        return [], list(sims_by_case), [], 0.0, 1.0
    return best


# ---------------------------------------------------------------------------
# Probe formatting.


def make_probe_id(prefix: str, text: str) -> str:
    digest = sha1(_norm(text).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def make_probe_label(text: str, max_len: int = 28) -> str:
    text = readable_text(text)
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip("，、；;。")


def make_question_seed(prototype_text: str, evidence_texts: Sequence[str] | None = None) -> str:
    evidence = [readable_text(x) for x in (evidence_texts or []) if readable_text(x)]
    if evidence:
        body = "；".join(evidence[:2])
    else:
        body = readable_text(prototype_text)
    return f"下面这些线索是否符合你的情况：{body}？请回答“是 / 不是 / 不确定”。"


# ---------------------------------------------------------------------------
# Broad anchor probes: good for upper tree levels and bilingual fields.


def _case_contains_anchor(case: CaseRecord | CaseCandidate, anchor: str) -> bool:
    return _norm(anchor) in _norm(case_search_text(case))


def _title_contains_anchor(case: CaseRecord | CaseCandidate, anchor: str) -> bool:
    title_bits = [case.title, *case.feature_tags, *_case_extra_texts(case)]
    return _norm(anchor) in _norm("\n".join(str(x) for x in title_bits if x))


def _evidence_for_anchor(case: CaseRecord | CaseCandidate, anchor: str) -> str:
    key = _norm(anchor)
    units = split_observable_units(case)
    for unit in sorted(units, key=len):
        if key and key in _norm(unit):
            return unit
    if key and key in _norm(case.description):
        return readable_text(case.description[:120])
    return readable_text(case.title)


def mine_anchor_probes(
    cases: Sequence[CaseRecord | CaseCandidate],
    *,
    asked_probe_ids: Iterable[str] | None = None,
    max_probes: int = 8,
    probe_prefix: str = "anchor",
    min_child_size: int = 1,
    min_child_ratio: float = 0.0,
) -> list[ProbeCandidate]:
    if len(cases) <= 1:
        return []

    total = len(cases)
    case_texts = {case.case_id: case_search_text(case) for case in cases}
    title_texts = {
        case.case_id: "\n".join([case.title, *case.feature_tags, *_case_extra_texts(case)])
        for case in cases
    }

    doc_freq: dict[str, set[str]] = {}
    title_freq: dict[str, set[str]] = {}
    for case in cases:
        terms = set(_candidate_terms(case_texts[case.case_id]))
        for term in terms:
            doc_freq.setdefault(term, set()).add(case.case_id)
        for term in set(_candidate_terms(title_texts[case.case_id])):
            title_freq.setdefault(term, set()).add(case.case_id)

    asked = set(asked_probe_ids or [])
    candidates: list[ProbeCandidate] = []
    min_child = max(1, min_child_size, int(total * min_child_ratio))

    for anchor, positive_set in doc_freq.items():
        pos_count = len(positive_set)
        neg_count = total - pos_count
        if pos_count < min_child or neg_count < min_child:
            continue
        if pos_count < 2:
            continue

        raw_score = split_quality(pos_count, neg_count, 0, total)
        title_count = len(title_freq.get(anchor, set()))
        title_ratio = title_count / max(1, pos_count)
        idf = math.log((1 + total) / (1 + pos_count)) + 1.0
        score = raw_score * (0.80 + 0.20 * min(idf, 2.2)) * (1.0 + 0.35 * title_ratio)
        if score <= 0.0:
            continue

        positive = [case.case_id for case in cases if case.case_id in positive_set]
        negative = [case.case_id for case in cases if case.case_id not in positive_set]
        evidence: list[str] = []
        for case in cases:
            if case.case_id in positive_set:
                ev = _evidence_for_anchor(case, anchor)
                if ev and ev not in evidence:
                    evidence.append(ev)
            if len(evidence) >= 5:
                break

        prototype = anchor
        probe_id = make_probe_id(probe_prefix, prototype)
        if probe_id in asked:
            continue

        candidates.append(
            ProbeCandidate(
                probe_id=probe_id,
                label=make_probe_label(anchor),
                prototype_text=prototype,
                question_seed=make_question_seed(prototype, evidence),
                positive_case_ids=positive,
                negative_case_ids=negative,
                unknown_case_ids=[],
                evidence_texts=evidence[:5],
                split_score=round(score, 4),
                debug={
                    "kind": "auto_anchor_bilingual",
                    "anchor": anchor,
                    "raw_split_score": round(raw_score, 4),
                    "positive": pos_count,
                    "negative": neg_count,
                    "unknown": 0,
                    "title_hits": title_count,
                    "idf": round(idf, 4),
                },
            )
        )

    candidates.sort(
        key=lambda p: (
            p.split_score,
            len(p.positive_case_ids) * len(p.negative_case_ids),
            len(p.evidence_texts),
        ),
        reverse=True,
    )
    return candidates[:max_probes]


# ---------------------------------------------------------------------------
# Local observation-window probes: useful inside narrower branches.


def _case_units_map(
    units: Sequence[TextUnit],
    idf: dict[str, float],
) -> dict[str, list[tuple[TextUnit, dict[str, float]]]]:
    out: dict[str, list[tuple[TextUnit, dict[str, float]]]] = {}
    for unit in units:
        out.setdefault(unit.case_id, []).append((unit, vectorize(unit.text, idf)))
    return out


def mine_local_probes(
    cases: Sequence[CaseRecord | CaseCandidate],
    *,
    asked_probe_ids: Iterable[str] | None = None,
    max_probes: int = 5,
    cluster_threshold: float = 0.52,
    min_score: float = 0.0,
    probe_prefix: str = "probe",
    min_child_size: int = 1,
    min_child_ratio: float = 0.0,
) -> list[ProbeCandidate]:
    case_ids = [case.case_id for case in cases]
    if len(case_ids) <= 1:
        return []

    units = collect_text_units(cases)
    if not units:
        return []

    idf = build_idf([unit.text for unit in units])
    unit_vecs = [(unit, vectorize(unit.text, idf)) for unit in units]
    asked = set(asked_probe_ids or [])

    clusters: list[dict[str, Any]] = []
    for unit, vec in unit_vecs:
        best_idx = -1
        best_sim = 0.0
        for idx, cluster in enumerate(clusters):
            sim = cosine(vec, cluster["centroid"])
            if sim > best_sim:
                best_sim = sim
                best_idx = idx
        if best_idx >= 0 and best_sim >= cluster_threshold:
            cluster = clusters[best_idx]
            cluster["members"].append((unit, vec))
            centroid: dict[str, float] = dict(cluster["centroid"])
            for key, value in vec.items():
                centroid[key] = centroid.get(key, 0.0) + value
            norm = math.sqrt(sum(v * v for v in centroid.values())) or 1.0
            cluster["centroid"] = {key: value / norm for key, value in centroid.items()}
        else:
            clusters.append({"members": [(unit, vec)], "centroid": dict(vec)})

    units_by_case = _case_units_map(units, idf)
    candidates: list[ProbeCandidate] = []

    for cluster_index, cluster in enumerate(clusters):
        members: list[tuple[TextUnit, dict[str, float]]] = cluster["members"]
        centroid: dict[str, float] = cluster["centroid"]
        if not members:
            continue

        prototype_unit, prototype_vec = max(members, key=lambda item: cosine(item[1], centroid))
        prototype_text = readable_text(prototype_unit.text)
        if not _is_observable_unit(prototype_text):
            continue

        probe_id = make_probe_id(probe_prefix, prototype_text)
        if probe_id in asked:
            continue

        sims: dict[str, float] = {}
        evidence_by_case: dict[str, str] = {}
        for cid in case_ids:
            best_unit_text = ""
            best_sim = 0.0
            for unit, vec in units_by_case.get(cid, []):
                sim = cosine(prototype_vec, vec)
                if sim > best_sim:
                    best_sim = sim
                    best_unit_text = unit.text
            sims[cid] = best_sim
            evidence_by_case[cid] = best_unit_text

        positive, negative, unknown, raw_score, threshold = _best_threshold_split(
            sims,
            min_child_size=min_child_size,
            min_child_ratio=min_child_ratio,
        )
        if raw_score < min_score:
            continue

        cluster_case_count = len({unit.case_id for unit, _ in members})
        coherence = sum(cosine(vec, centroid) for _, vec in members) / max(1, len(members))
        multi_case_bonus = min(1.0, cluster_case_count / max(2, min(6, len(case_ids))))
        final_score = raw_score * (0.75 + 0.25 * coherence) * (0.75 + 0.25 * multi_case_bonus)
        if final_score < min_score:
            continue

        evidence_texts: list[str] = []
        for cid in positive[:8]:
            ev = evidence_by_case.get(cid)
            if ev and ev not in evidence_texts:
                evidence_texts.append(ev)

        candidates.append(
            ProbeCandidate(
                probe_id=probe_id,
                label=make_probe_label(prototype_text),
                prototype_text=prototype_text,
                question_seed=make_question_seed(prototype_text, evidence_texts),
                positive_case_ids=positive,
                negative_case_ids=negative,
                unknown_case_ids=unknown,
                evidence_texts=evidence_texts[:5],
                split_score=round(final_score, 4),
                debug={
                    "kind": "local_window_cluster",
                    "raw_split_score": round(raw_score, 4),
                    "threshold": round(threshold, 4),
                    "positive": len(positive),
                    "negative": len(negative),
                    "unknown": len(unknown),
                    "cluster_cases": cluster_case_count,
                    "coherence": round(coherence, 4),
                    "cluster_index": cluster_index,
                },
            )
        )

    candidates.sort(key=lambda p: (p.split_score, -len(p.unknown_case_ids), len(p.evidence_texts)), reverse=True)
    return candidates[:max_probes]


def mine_tree_probes(
    cases: Sequence[CaseRecord | CaseCandidate],
    *,
    asked_probe_ids: Iterable[str] | None = None,
    max_probes: int = 5,
    probe_prefix: str = "probe",
    min_child_size: int = 1,
    min_child_ratio: float = 0.0,
) -> list[ProbeCandidate]:
    """Return the best broad and local probes for a tree node."""

    broad = mine_anchor_probes(
        cases,
        asked_probe_ids=asked_probe_ids,
        max_probes=max(max_probes * 2, 8),
        probe_prefix=f"{probe_prefix}_a",
        min_child_size=min_child_size,
        min_child_ratio=min_child_ratio,
    )
    local = mine_local_probes(
        cases,
        asked_probe_ids=asked_probe_ids,
        max_probes=max(max_probes * 2, 8),
        min_score=0.0,
        probe_prefix=f"{probe_prefix}_l",
        min_child_size=min_child_size,
        min_child_ratio=min_child_ratio,
    )

    merged: list[ProbeCandidate] = []
    seen_signatures: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
    for probe in sorted([*broad, *local], key=lambda p: p.split_score, reverse=True):
        sig = (tuple(sorted(probe.positive_case_ids)), tuple(sorted(probe.negative_case_ids)))
        if sig in seen_signatures:
            continue
        seen_signatures.add(sig)
        merged.append(probe)
        if len(merged) >= max_probes:
            break
    return merged


# ---------------------------------------------------------------------------
# Answer/search utility compatibility.


def classify_answer(text: str, feature_id: str | None = None) -> str:
    """Compatibility wrapper. Prefer answer_parser.parse_probe_answer_signal."""

    from medical_assistant.services.cases.answer_parser import parse_probe_answer_signal

    return parse_probe_answer_signal(
        question_text="",
        user_answer=text,
        probe_label=feature_id or "",
    )


def extract_search_terms(text: str) -> list[str]:
    text = readable_text(text)
    if not text:
        return []
    chunks = [readable_text(x) for x in re.split(r"[，,。；;！!？?、\s]+", text) if readable_text(x)]
    terms: list[str] = []
    for chunk in chunks:
        terms.append(chunk)
        compact = _norm(chunk)
        if len(compact) >= 4:
            for n in (2, 3):
                for i in range(0, min(len(compact) - n + 1, 6)):
                    terms.append(compact[i : i + n])
        terms.extend(_latin_terms(chunk))
    return _dedupe_keep_order(terms)[:16]
