from __future__ import annotations

from functools import lru_cache
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from medical_assistant.graph.nodes.answer import answer_node
from medical_assistant.graph.nodes.answer import safety_response_node
from medical_assistant.graph.nodes.assess_safety import assess_safety_node
from medical_assistant.graph.nodes.clarify import clarify_node
from medical_assistant.graph.nodes.estimate_confidence import estimate_confidence_node
from medical_assistant.graph.nodes.merge_candidates import merge_candidates_node
from medical_assistant.graph.nodes.normalize import normalize_node
from medical_assistant.graph.nodes.retrieve_local import retrieve_local_node
from medical_assistant.graph.nodes.retrieve_pubmed import retrieve_pubmed_node
from medical_assistant.graph.nodes.update_context import update_context_node
from medical_assistant.graph.routes import route_after_confidence


class WorkflowState(TypedDict, total=False):
    question: str
    conversation_state: dict[str, Any]
    normalized: dict[str, Any]
    local_result: dict[str, Any]
    candidate_topics: list[dict[str, Any]]
    safety: dict[str, Any]
    confidence: dict[str, Any]
    web_result: dict[str, Any]
    response: dict[str, Any]


@lru_cache(maxsize=1)
def build_workflow():
    graph = StateGraph(WorkflowState)

    graph.add_node("normalize", normalize_node)
    graph.add_node("update_context", update_context_node)
    graph.add_node("retrieve_local", retrieve_local_node)
    graph.add_node("merge_candidates", merge_candidates_node)
    graph.add_node("assess_safety", assess_safety_node)
    graph.add_node("estimate_confidence", estimate_confidence_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("retrieve_pubmed", retrieve_pubmed_node)
    graph.add_node("answer", answer_node)
    graph.add_node("safety_response", safety_response_node)

    graph.add_edge(START, "normalize")
    graph.add_edge("normalize", "update_context")
    graph.add_edge("update_context", "retrieve_local")
    graph.add_edge("retrieve_local", "merge_candidates")
    graph.add_edge("merge_candidates", "assess_safety")
    graph.add_edge("assess_safety", "estimate_confidence")
    graph.add_conditional_edges(
        "estimate_confidence",
        route_after_confidence,
        {
            "clarify": "clarify",
            "answer": "answer",
            "search_pubmed": "retrieve_pubmed",
            "safety": "safety_response",
        },
    )
    graph.add_edge("retrieve_pubmed", "answer")
    graph.add_edge("clarify", END)
    graph.add_edge("answer", END)
    graph.add_edge("safety_response", END)

    return graph.compile()
