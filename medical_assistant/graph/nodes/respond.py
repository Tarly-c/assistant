"""Compatibility wrappers for the old RAG nodes.

The stable demo workflow uses plan_question.py and final_answer.py directly.
"""
from __future__ import annotations

from medical_assistant.graph.nodes.final_answer import final_answer_node as answer_node
from medical_assistant.graph.nodes.plan_question import plan_question_node as clarify_node
from medical_assistant.graph.nodes.plan_question import route_after_cases as route_after_retrieve
