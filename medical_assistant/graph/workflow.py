"""构建问诊 workflow。"""
from __future__ import annotations

from medical_assistant.graph.nodes import (
    parse_input, narrow, score, ask, answer, route,
)
from medical_assistant.graph.state import S

try:
    from langgraph.graph import END, StateGraph
except Exception:
    END = StateGraph = None


class _Fallback:
    """无 LangGraph 时的顺序执行。"""
    def invoke(self, state: S) -> S:
        s: S = dict(state)
        for fn in (parse_input, narrow, score):
            s.update(fn(s))
        s.update(answer(s) if route(s) == "answer" else ask(s))
        return s


def build_workflow():
    if StateGraph is None:
        return _Fallback()
    g = StateGraph(S)
    g.add_node("parse_input", parse_input)
    g.add_node("narrow", narrow)
    g.add_node("score", score)
    g.add_node("ask", ask)
    g.add_node("answer", answer)
    g.set_entry_point("parse_input")
    g.add_edge("parse_input", "narrow")
    g.add_edge("narrow", "score")
    g.add_conditional_edges("score", route, {"ask": "ask", "answer": "answer"})
    g.add_edge("ask", END)
    g.add_edge("answer", END)
    return g.compile()
