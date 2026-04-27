"""Prompt 模板。"""

# 从用户输入中抽取双语检索信息 + 关键概念
NORMALIZE = """\
你是医学问题分析助手。请从用户的中文医学问题中抽取：

- query_cn: 中文精简查询（去语气词，保留核心症状）
- query_en: 对应的英文检索查询
- intent: treatment / cause / symptom / diagnosis / general
- concepts: 关键医学概念列表，每个包含：
  - term: 概念词（2-8字）
  - role: 它在用户描述中的角色（自由描述，5-15字，如"疼痛发生的位置""引发不适的外界因素"）
"""

# 离线：从病例中抽取结构化概念
EXTRACT_CONCEPTS = """\
你是医学信息抽取专家。请从下面的病例中提取所有有临床区分意义的关键概念。

对每个概念，输出：
- term: 概念本身（2-8字）
- role: 它在这个病例中扮演什么角色（自由描述，5-15字，如"引发疼痛的外界因素""患牙所在区域"）
- importance: 区分重要性 high / medium / low
- negative: 是否为"本病例明确不会出现的"（true/false，默认 false）

注意：
1. 不需要遵循固定分类体系，用你自己的话描述 role
2. 同时提取"有的"和"明确没有的"特征
3. 提取 5-12 个最有区分力的概念

病例标题：{title}
病例描述：{description}
"""

# 给概念簇命名
NAME_CLUSTER = """\
下面是一组医学概念在病例中的角色描述：
{roles}

请用一个简短的中文标签（2-6字）概括这组概念共同的角色类型。
只输出标签，不输出其他内容。
"""

# 判断用户回答
PARSE_ANSWER = """\
你是问诊回答解析器。判断用户这句话是否回答了上一轮追问。
只输出 signal / confidence / evidence / new_observations。
signal 只能是 yes / no / uncertain / unrelated。
"""

# LLM 改写追问
REPHRASE_PROBE = """\
你是口腔科问诊医生。请把下面的技术描述改写成一个自然、口语化的中文追问。
要求：
1. 用患者能听懂的日常用语
2. 只问一个维度，不堆砌
3. 以问号结尾，20-40 字左右

技术描述：{description}
支撑证据：{evidence}

请直接输出改写后的追问。
"""
