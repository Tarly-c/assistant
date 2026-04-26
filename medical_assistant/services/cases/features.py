from __future__ import annotations

"""Data-driven dynamic feature/probe utilities for case localization.

The module does not keep a hand-written symptom table, authoring-token filter,
discourse-prefix table, or yes/no vocabulary. It mines patient-observable probes
from the current case texts and scores how well each probe splits a candidate
solution set. The same logic is used by the offline question-tree builder and
by the online local fallback planner.
"""

from dataclasses import dataclass, field
from hashlib import sha1
import math
import re
from typing import Iterable, Sequence

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
    debug: dict[str, float | int | str] = field(default_factory=dict)

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
            split_score=round(self.split_score, 4),
            strategy=strategy,
            tree_node_id=tree_node_id,
            yes_child_id=yes_child_id,
            no_child_id=no_child_id,
            evidence_texts=self.evidence_texts[:5],
            debug=self.debug,
        )


# Compatibility shims for older imports. The old code used a static feature
# bank. New code should not depend on global feature ids.
def all_features() -> tuple[object, ...]:
    return ()


def get_feature(feature_id: str) -> None:
    return None


def extract_features(text: str) -> list[str]:
    return []


def feature_labels(feature_ids: Iterable[str]) -> list[str]:
    return [str(fid) for fid in feature_ids if fid]


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").strip().lower())


def readable_text(text: str) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    return text.strip(" ，,。.;；、")


def _content_score(text: str) -> int:
    compact = _norm(text)
    # Generic content measure: count CJK characters and alphanumerics. This is
    # not domain vocabulary; it only prevents empty punctuation fragments.
    return len(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", compact))


def _is_observable_unit(text: str) -> bool:
    text = readable_text(text)
    compact = _norm(text)
    if not compact:
        return False
    if len(compact) < 5 or len(compact) > 160:
        return False
    return _content_score(compact) >= 5


def split_observation_units(
    title: str = "",
    description: str = "",
    *,
    include_title: bool = False,
    max_window: int = 3,
) -> list[str]:
    """Split case text into context-preserving observation windows.

    The function deliberately does not rely on a phrase list such as dependent
    prefixes. Instead it creates adjacent one-, two-, and three-clause windows.
    This keeps orphan clauses attached to their local context without encoding
    Chinese discourse examples in code.
    """

    raw = description or ""
    clauses = [
        readable_text(x)
        for x in re.split(r"[。；;！!？?\n]+", raw)
        if readable_text(x)
    ]

    units: list[str] = []
    if include_title and title:
        title_text = readable_text(title)
        if _is_observable_unit(title_text):
            units.append(title_text)

    for i in range(len(clauses)):
        for size in range(1, max_window + 1):
            window_clauses = clauses[i : i + size]
            if len(window_clauses) != size:
                continue
            window = readable_text("，".join(window_clauses))
            if _is_observable_unit(window):
                units.append(window)

    seen: set[str] = set()
    out: list[str] = []
    for unit in units:
        key = _norm(unit)
        if key and key not in seen:
            out.append(unit)
            seen.add(key)
    return out


def collect_text_units(cases: Sequence[CaseRecord | CaseCandidate]) -> list[TextUnit]:
    units: list[TextUnit] = []
    for case in cases:
        for text in split_observation_units(case.title, case.description):
            units.append(TextUnit(case_id=case.case_id, text=text))
    return units


def _char_ngrams(text: str, ns: tuple[int, ...] = (2, 3, 4)) -> list[str]:
    compact = _norm(text)
    if not compact:
        return []
    grams: list[str] = []
    for n in ns:
        if len(compact) >= n:
            grams.extend(compact[i : i + n] for i in range(len(compact) - n + 1))
    grams.extend(re.findall(r"[a-z0-9_]{2,}", compact))
    return grams


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
    return max(0.0, entropy * coverage * (balance ** 1.35) * unknown_penalty)


def _best_threshold_split(
    sims_by_case: dict[str, float],
    *,
    margin: float = 0.035,
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
        if threshold < 0.02:
            continue
        positive = [cid for cid, sim in sims_by_case.items() if sim >= threshold + margin]
        unknown = [cid for cid, sim in sims_by_case.items() if threshold - margin <= sim < threshold + margin]
        negative = [cid for cid, sim in sims_by_case.items() if sim < threshold - margin]
        ratio_child = int(total * min_child_ratio)
        min_child = max(1, min_child_size, ratio_child)
        if len(positive) < min_child or len(negative) < min_child:
            continue
        score = split_quality(len(positive), len(negative), len(unknown), total)
        if score > best_score:
            best_score = score
            best = (positive, negative, unknown, score, threshold)

    if best is None:
        return [], list(sims_by_case), [], 0.0, 1.0
    return best


def make_probe_id(prefix: str, text: str) -> str:
    digest = sha1(_norm(text).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def make_probe_label(text: str, max_len: int = 28) -> str:
    text = readable_text(text)
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip("，、；;。")


def make_question_seed(prototype_text: str) -> str:
    text = readable_text(prototype_text)
    return f"是否有这种情况：{text}？请回答：是 / 不是 / 不确定。"


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
    cluster_threshold: float = 0.46,
    min_score: float = 0.08,
    probe_prefix: str = "probe",
    min_child_size: int = 1,
    min_child_ratio: float = 0.0,
) -> list[ProbeCandidate]:
    """Mine discriminative probes from the provided case subset.

    It uses TF-IDF character n-gram vectors as a lightweight deterministic
    stand-in for external embeddings, so offline tree building can run in local
    demos without an embedding service.
    """

    case_ids = [case.case_id for case in cases]
    if len(case_ids) <= 1:
        return []

    units = collect_text_units(cases)
    if not units:
        return []

    idf = build_idf([unit.text for unit in units])
    unit_vecs = [(unit, vectorize(unit.text, idf)) for unit in units]
    asked = set(asked_probe_ids or [])

    clusters: list[dict[str, object]] = []
    for unit, vec in unit_vecs:
        best_idx = -1
        best_sim = 0.0
        for idx, cluster in enumerate(clusters):
            sim = cosine(vec, cluster["centroid"])  # type: ignore[arg-type]
            if sim > best_sim:
                best_sim = sim
                best_idx = idx
        if best_idx >= 0 and best_sim >= cluster_threshold:
            cluster = clusters[best_idx]
            members = cluster["members"]  # type: ignore[assignment]
            members.append((unit, vec))  # type: ignore[union-attr]
            centroid: dict[str, float] = dict(cluster["centroid"])  # type: ignore[arg-type]
            for key, value in vec.items():
                centroid[key] = centroid.get(key, 0.0) + value
            norm = math.sqrt(sum(v * v for v in centroid.values())) or 1.0
            cluster["centroid"] = {key: value / norm for key, value in centroid.items()}
        else:
            clusters.append({"members": [(unit, vec)], "centroid": dict(vec)})

    units_by_case = _case_units_map(units, idf)
    candidates: list[ProbeCandidate] = []

    for cluster_index, cluster in enumerate(clusters):
        members: list[tuple[TextUnit, dict[str, float]]] = cluster["members"]  # type: ignore[assignment]
        centroid: dict[str, float] = cluster["centroid"]  # type: ignore[assignment]
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

        positive, negative, unknown, split_score, threshold = _best_threshold_split(
            sims,
            min_child_size=min_child_size,
            min_child_ratio=min_child_ratio,
        )
        if split_score < min_score:
            continue

        cluster_case_count = len({unit.case_id for unit, _ in members})
        coherence = sum(cosine(vec, centroid) for _, vec in members) / max(1, len(members))
        multi_case_bonus = min(1.0, cluster_case_count / max(2, min(6, len(case_ids))))
        final_score = split_score * (0.75 + 0.25 * coherence) * (0.75 + 0.25 * multi_case_bonus)
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
                question_seed=make_question_seed(prototype_text),
                positive_case_ids=positive,
                negative_case_ids=negative,
                unknown_case_ids=unknown,
                evidence_texts=evidence_texts,
                split_score=round(final_score, 4),
                debug={
                    "raw_split_score": round(split_score, 4),
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


def extract_search_terms(text: str) -> list[str]:
    """Extract generic retrieval chunks from user text without static vocabulary."""

    text = readable_text(text)
    if not text:
        return []

    chunks = [
        readable_text(x)
        for x in re.split(r"[，,。；;！!？?、\s]+", text)
        if readable_text(x)
    ]
    terms: list[str] = []
    for chunk in chunks:
        compact = _norm(chunk)
        if len(compact) < 2:
            continue
        terms.append(chunk)
        if len(compact) >= 4:
            for n in (2, 3):
                limit = min(len(compact) - n + 1, 4)
                for i in range(max(0, limit)):
                    terms.append(compact[i : i + n])

    seen: set[str] = set()
    out: list[str] = []
    for term in terms:
        if term and term not in seen:
            out.append(term)
            seen.add(term)
    return out[:12]
