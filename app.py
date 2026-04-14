from __future__ import annotations

import json
import os
from typing import Literal, TypedDict

from pydantic import BaseModel, Field

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from kb import search_local
from pubmed import search_pubmed_best


CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:7b")


class SearchPlan(BaseModel):
    local_query: str = Field(description="给本地知识库用的短查询")
    pubmed_queries: list[str] = Field(
        default_factory=list,
        description="3 个左右英文 PubMed 概念查询",
    )
    intent: Literal["treatment", "cause", "symptom", "diagnosis", "general"] = "general"


class AgentState(TypedDict, total=False):
    question: str
    plan: dict
    local_result: dict
    web_result: dict
    answer: str


llm = ChatOllama(model=CHAT_MODEL, temperature=0)
planner = llm.with_structured_output(SearchPlan)


PLAN_PROMPT = """
把用户的医学问题改写成一个检索计划。

要求：
1. local_query：给本地知识库做语义检索，尽量短。
2. pubmed_queries：给 PubMed 用的英文概念查询，输出 3 个左右。
3. 对“吃什么药 / 怎么治 / 怎么办”这类问题，优先生成“疾病或症状 + treatment/management/review/guideline”。
4. 避免只输出像 "headache medication" 这种容易漂到 medication overuse headache 的查询。
5. 只返回结构化结果，不要解释。

示例：
问题：头痛吃什么药
local_query：头痛 对症处理
pubmed_queries：
- headache treatment review
- acute headache treatment adults
- migraine acute treatment guideline
""".strip()


ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
你是医学信息整理助手。
只能基于给定证据回答，不要编造。
这不是个体化诊断，也不是处方建议。

回答格式固定为四段：
1. 结论
2. 依据
3. 来源
4. 风险提示 / 何时就医

要求：
- 如果证据不足，明确写“当前证据不足”。
- 如果是常见症状的一般性处理建议，要和“处方/个体化用药建议”分开。
- 来源段尽量引用 [Local 1] / [PubMed 1] 这样的标签。
""".strip(),
        ),
        (
            "human",
            """
用户问题：
{question}

证据：
{evidence}
""".strip(),
        ),
    ]
)


def plan_query(state: AgentState) -> AgentState:
    plan = planner.invoke(
        [
            {"role": "system", "content": PLAN_PROMPT},
            {"role": "user", "content": state["question"]},
        ]
    )
    return {"plan": plan.model_dump()}


def search_local_node(state: AgentState) -> AgentState:
    plan = state["plan"]
    local_result = search_local(
        question=state["question"],
        local_query=plan["local_query"],
    )
    return {"local_result": local_result}


def route_after_local(state: AgentState) -> str:
    local_result = state.get("local_result", {})
    return "answer" if local_result.get("enough") else "search_pubmed"


def search_pubmed_node(state: AgentState) -> AgentState:
    web_result = search_pubmed_best(
        question=state["question"],
        queries=state["plan"].get("pubmed_queries", []),
    )
    return {"web_result": web_result}


def _render_evidence(state: AgentState) -> str:
    parts: list[str] = []

    local_result = state.get("local_result", {})
    for i, hit in enumerate(local_result.get("hits", []), start=1):
        parts.append(
            f"[Local {i}] source={hit.get('source')} chunk={hit.get('chunk_id')} "
            f"score={hit.get('score')}\n{hit.get('snippet')}"
        )

    web_result = state.get("web_result", {})
    for i, hit in enumerate(web_result.get("hits", []), start=1):
        parts.append(
            f"[PubMed {i}] PMID={hit.get('pmid')} title={hit.get('title')} "
            f"journal={hit.get('journal')} pubdate={hit.get('pubdate')} "
            f"rerank_score={hit.get('rerank_score')}\n{hit.get('snippet')}"
        )

    if not parts:
        parts.append("无可用证据。")

    return "\n\n".join(parts)


def answer_node(state: AgentState) -> AgentState:
    chain = ANSWER_PROMPT | llm | StrOutputParser()
    answer = chain.invoke(
        {
            "question": state["question"],
            "evidence": _render_evidence(state),
        }
    )
    return {"answer": answer}


graph = StateGraph(AgentState)
graph.add_node("plan_query", plan_query)
graph.add_node("search_local", search_local_node)
graph.add_node("search_pubmed", search_pubmed_node)
graph.add_node("answer", answer_node)

graph.add_edge(START, "plan_query")
graph.add_edge("plan_query", "search_local")
graph.add_conditional_edges(
    "search_local",
    route_after_local,
    {
        "search_pubmed": "search_pubmed",
        "answer": "answer",
    },
)
graph.add_edge("search_pubmed", "answer")
graph.add_edge("answer", END)

app = graph.compile()


def main() -> None:
    print("Medical LangGraph Assistant 已就绪。输入 exit 退出。")

    while True:
        try:
            question = input("\n[User]> ").strip()
            if not question:
                continue
            if question.lower() == "exit":
                break

            result = app.invoke({"question": question})

            plan = result.get("plan", {})
            local_result = result.get("local_result", {})
            web_result = result.get("web_result", {})

            print("\n[Plan]", json.dumps(plan, ensure_ascii=False))
            print("[Local]", json.dumps(local_result, ensure_ascii=False))
            if web_result:
                print("[PubMed]", json.dumps(web_result, ensure_ascii=False))

            print(f"\n[Agent]> {result['answer']}")
        except KeyboardInterrupt:
            print("\n安全退出。")
            break
        except Exception as exc:
            print(f"\n[Agent]> 系统错误: {exc}")


if __name__ == "__main__":
    main()
