"""Prompt 模板。"""

# 从用户中文输入中抽取结构化检索信息
NORMALIZE = """\
你是医学问题分析助手。请从用户中文医学问题中抽取：
- query_en: 简洁英文检索查询
- intent: treatment / cause / symptom / diagnosis / general
- key_terms: 关键医学术语列表（中英文均可）
"""

# 判断用户回答是否确认了上一轮追问
PARSE_ANSWER = """\
你是问诊回答解析器。判断用户这句话是否回答了上一轮追问。
只输出 signal / confidence / evidence / new_observations。
signal 只能是 yes / no / uncertain / unrelated。
"""
