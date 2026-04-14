import os
from pydantic import BaseModel, Field

# 1. 业务逻辑：极简版倒排索引/字符串匹配检索
def search_local_papers(keywords: str) -> str:
    """在本地 resource 文件夹中检索医学论文"""
    folder = "resources"
    print(f"📂 [本地检索] 正在资料库检索: {keywords}")
    
    if not os.path.exists(folder):
        return "LOCAL_NOT_FOUND: 本地资源库目录不存在，请尝试网络检索。"
        
    results = []
    # 遍历本地文件
    for filename in os.listdir(folder):
        if filename.endswith(".txt") or filename.endswith(".md"):
            filepath = os.path.join(folder, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                # 忽略大小写的简单关键词匹配
                if keywords.lower() in content.lower():
                    # 提取关键词前后的上下文片段 (模拟 RAG 的 Chunking)
                    idx = content.lower().find(keywords.lower())
                    start = max(0, idx - 150)
                    end = min(len(content), idx + 300)
                    snippet = content[start:end].replace('\n', ' ')
                    results.append(f"【来源: {filename}】\n...{snippet}...")
                    
    if results:
        # 限制返回长度，防止爆 Token
        final_result = "\n\n".join(results)[:2000] 
        return f"找到本地医学资料：\n{final_result}"
        
    # 核心机制：明确告诉大模型本地没查到
    return "LOCAL_NOT_FOUND: 本地资料库未找到相关论文，请立即调用 web_medical_search 进行网络检索。"

# 2. 参数校验模型
class LocalSearchArgs(BaseModel):
    keywords: str = Field(..., description="要检索的医学核心关键词，例如 '阿司匹林 心血管'")

# 3. 技能契约
SKILL_DEF = {
    "name": "search_local_papers",
    "description": "优先使用此工具检索本地医学资料库中的专业论文。返回 LOCAL_NOT_FOUND 时说明本地无数据。",
    "func": search_local_papers,
    "args_schema": LocalSearchArgs
}