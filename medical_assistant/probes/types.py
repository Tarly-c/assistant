"""Probe 数据类型。"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from medical_assistant.schemas import Probe


@dataclass
class ProbeCandidate:
    """挖掘出的候选 probe。"""
    probe_id: str
    feature_dim: int                  # 对应 (K+M) 特征空间的维度索引
    label: str
    question: str                     # LLM 改写后的追问
    positive_ids: list[str]
    negative_ids: list[str]
    unknown_ids: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    score: float = 0.0
    debug: dict[str, Any] = field(default_factory=dict)

    def to_probe(self, *, strategy: str = "dynamic",
                 tree_node: str = "", yes_child: str = "", no_child: str = "") -> Probe:
        return Probe(
            probe_id=self.probe_id, feature_dim=self.feature_dim,
            label=self.label, text=self.question,
            positive_ids=self.positive_ids, negative_ids=self.negative_ids,
            unknown_ids=self.unknown_ids, score=round(self.score, 4),
            strategy=strategy, tree_node=tree_node,
            yes_child=yes_child, no_child=no_child,
            evidence=self.evidence[:5], debug=self.debug,
        )
