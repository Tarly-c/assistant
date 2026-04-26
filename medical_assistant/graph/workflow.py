"""构建病例定位 workflow。"""
from __future__ import annotations

from medical_assistant.graph.nodes import (
    normalize_node, update_memory_node, narrow_cases_node,
    retrieve_cases_node, plan_question_node, final_answer_node,
    route_after_cases,
)
from medical_assistant.graph.state import GraphState

try:
    from langgraph.graph import END, StateGraph
except Exception:
    END = StateGraph = None


class _Fallback:
    def invoke(self, state: GraphState) -> GraphState:
        s: GraphState = dict(state)
        for fn in (normalize_node, update_memory_node, narrow_cases_node, retrieve_cases_node):
            s.update(fn(s))
        if route_after_cases(s) == "final_answer":
            s.update(final_answer_node(s))
        else:
            s.update(plan_question_node(s))
        return s


def build_workflow():
    if StateGraph is None:
        return _Fallback()
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
    g.add_conditional_edges("retrieve_cases", route_after_cases,
                            {"plan_question": "plan_question", "final_answer": "final_answer"})
    g.add_edge("plan_question", END)
    g.add_edge("final_answer", END)
    return g.compile()
