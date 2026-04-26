"""Probe 相关的数据类型。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from medical_assistant.schemas import PlannedQuestion


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
        self, *,
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
