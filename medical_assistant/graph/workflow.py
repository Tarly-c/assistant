"""构建简化版 LangGraph 工作流 — 无 safety。"""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from medical_assistant.graph.state import GraphState
from medical_assistant.graph.nodes.normalize import normalize_node
from medical_assistant.graph.nodes.retrieve import retrieve_node
from medical_assistant.graph.nodes.respond import (
    answer_node,
    clarify_node,
    route_after_retrieve,
)


def build_workflow():
    g = StateGraph(GraphState)

    g.add_node("normalize", normalize_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("answer", answer_node)
    g.add_node("clarify", clarify_node)

    g.set_entry_point("normalize")
    g.add_edge("normalize", "retrieve")

    g.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {
            "answer": "answer",
            "clarify": "clarify",
        },
    )

    g.add_edge("answer", END)
    g.add_edge("clarify", END)

    return g.compile()
