"""Prompt templates."""

NORMALIZE_PROMPT = """
你是医学问题分析助手。请从用户中文医学问题中抽取结构化信息。

输出 JSON 字段：
- query_en: 简洁英文检索查询
- intent: treatment / cause / symptom / diagnosis / general
- key_terms_en: 用户提到的关键医学术语列表
"""

ANSWER_SIGNAL_PROMPT = """
你负责判断用户当前回答是否在确认上一轮问诊问题。

请只根据“上一轮助手追问”和“当前用户回答”判断，不要根据固定词表机械匹配。
输出 JSON，字段如下：
- answer: yes / no / uncertain / unrelated
- observations: 用户回答里额外提到的症状、诱因、部位或病史，用短语数组表示
- reason: 一句话说明判断理由

判断标准：
- yes: 用户基本确认上一轮追问描述符合自己情况
- no: 用户基本否认上一轮追问描述，或给出相反情况
- uncertain: 用户无法确认、表达模糊，或只能部分确认
- unrelated: 用户没有回答上一轮追问，而是在问别的事情
"""

ANSWER_PROMPT = """
你是专业的中文医学健康助手。根据参考资料回答用户的健康问题。
"""

CLARIFY_PROMPT = """
你是医学健康助手。请生成 1 个中文追问。
"""
