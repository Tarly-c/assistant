from __future__ import annotations

from medical_assistant.schemas.input import ConceptCandidate, Facet, NormalizedInput
from medical_assistant.services.terminology.local_terms import (
    detect_chief_complaint,
    detect_findings,
    detect_negated_findings,
    detect_red_flags,
)
from medical_assistant.services.terminology.mesh import lookup_mesh
from medical_assistant.services.terminology.aliases import LOCAL_STOP_TERMS


def _unique(values: list[str], limit: int | None = None) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = (item or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
        if limit and len(result) >= limit:
            break
    return result


def resolve_terminology(text: str, current_terms: list[str] | None = None) -> NormalizedInput:
    current_terms = current_terms or []

    chief = detect_chief_complaint(text)
    findings = detect_findings(text)
    negated = detect_negated_findings(text)
    red_flags = detect_red_flags(text)

    normalized_terms = list(current_terms)
    candidate_concepts: list[ConceptCandidate] = []
    facets: list[Facet] = []
    negated_findings: list[Facet] = []

    if chief:
        normalized_terms.append(chief["english"])
        candidate_concepts.append(
            ConceptCandidate(
                term=chief["english"],
                support=0.65,
                source="dictionary",
            )
        )
        mesh_matches = lookup_mesh(chief["english"], limit=2)
        for item in mesh_matches:
            candidate_concepts.append(
                ConceptCandidate(
                    term=item["label"],
                    concept_id=item.get("id"),
                    support=0.72,
                    source="dictionary",
                )
            )

    for item in findings:
        if item.get("normalized_term"):
            normalized_terms.append(item["normalized_term"])
        facets.append(
            Facet(
                name=item["name"],
                value=item.get("value"),
                status="present",
                source="dictionary",
                confidence=0.82,
            )
        )

    for item in negated:
        negated_findings.append(
            Facet(
                name=item["name"],
                value=item.get("value"),
                status="negated",
                source="dictionary",
                confidence=0.9,
            )
        )

    unresolved: list[str] = []
    if chief and chief["english"] == "headache":
        known_names = {item.name for item in facets + negated_findings}
        if "sudden_onset" not in known_names:
            unresolved.append("sudden_onset")
        if "pain_quality" not in known_names:
            unresolved.append("pain_quality")
    elif chief and chief["english"] == "cough":
        known_names = {item.name for item in facets + negated_findings}
        if "duration" not in known_names:
            unresolved.append("duration")
        if "shortness_of_breath" not in known_names:
            unresolved.append("shortness_of_breath")

    normalized_terms = [
        term for term in _unique(normalized_terms, limit=8)
        if term.lower() not in LOCAL_STOP_TERMS
    ]

    return NormalizedInput(
        original_text=text,
        chief_complaint=chief["english"] if chief else None,
        normalized_terms=normalized_terms,
        search_keywords=normalized_terms,
        candidate_concepts=candidate_concepts[:5],
        facets=facets,
        negated_findings=negated_findings,
        red_flags=_unique(red_flags, limit=8),
        unresolved_questions=_unique(unresolved, limit=3),
        free_text_observations=["规则词典已补充术语映射。"],
    )
