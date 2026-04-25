"""构建稳定的病例定位 workflow。

当前 demo 只使用本地病例库，不接 web_search，也不加 safety 节点。
如果本地还没安装 langgraph，会自动使用同等顺序的轻量 fallback，方便先跑 demo。
"""
from __future__ import annotations

from medical_assistant.graph.nodes.final_answer import final_answer_node
from medical_assistant.graph.nodes.narrow_cases import narrow_cases_node
from medical_assistant.graph.nodes.normalize import normalize_node
from medical_assistant.graph.nodes.plan_question import plan_question_node, route_after_cases
from medical_assistant.graph.nodes.retrieve_cases import retrieve_cases_node
from medical_assistant.graph.nodes.update_memory import update_memory_node
from medical_assistant.graph.state import GraphState

try:  # langgraph is still the intended runtime dependency.
    from langgraph.graph import END, StateGraph
except Exception:  # pragma: no cover - depends on local environment
    END = None
    StateGraph = None


class SequentialCaseWorkflow:
    """Small fallback with the same node order as the LangGraph version."""

    def invoke(self, state: GraphState) -> GraphState:
        merged: GraphState = dict(state)
        for node in (normalize_node, update_memory_node, narrow_cases_node, retrieve_cases_node):
            merged.update(node(merged))
        if route_after_cases(merged) == "final_answer":
            merged.update(final_answer_node(merged))
        else:
            merged.update(plan_question_node(merged))
        return merged


def build_workflow():
    if StateGraph is None:
        return SequentialCaseWorkflow()

    g = StateGraph(GraphState)

    g.add_node("normalize", normalize_node)
    g.add_node("update_memory", update_memory_node)
    g.add_node("narrow_cases", narrow_cases_node)
    g.add_node("retrieve_cases", retrieve_cases_node)
    g.add_node("plan_question", plan_question_node)
    g.add_node("final_answer", final_answer_node)

    g.set_entry_point("normalize")
    g.add_edge("normalize", "update_memory")
    g.add_edge("update_memory", "narrow_cases")
    g.add_edge("narrow_cases", "retrieve_cases")
    g.add_conditional_edges(
        "retrieve_cases",
        route_after_cases,
        {
            "plan_question": "plan_question",
            "final_answer": "final_answer",
        },
    )
    g.add_edge("plan_question", END)
    g.add_edge("final_answer", END)

    return g.compile()
