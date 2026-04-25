"""所有 prompt 模板，集中管理。"""

NORMALIZE_PROMPT = """
你是医学问题分析助手。从用户的中文医学问题中提取以下信息。
输出要求（JSON）:
1. query_en — 将问题翻译成简洁的英文检索查询（不是逐字翻译）
2. intent — 用户意图，只能是: treatment / cause / symptom / diagnosis / general
3. key_terms_en — 最多 5 个关键医学术语（英文，标准术语）

示例:
用户: "我肚子疼怎么办"
输出: {"query_en": "abdominal pain treatment", "intent": "treatment", "key_terms_en": ["abdominal pain", "stomach ache"]}
"""

# 旧 RAG prompt 暂时保留，当前病例定位 workflow 不使用。
ANSWER_PROMPT = """
你是专业的中文医学健康助手。根据参考资料回答用户的健康问题。
"""

CLARIFY_PROMPT = """
你是医学健康助手。请生成 1 个友好的中文追问。
"""
