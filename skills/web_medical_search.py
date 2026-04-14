import json
import urllib.request
import urllib.parse
from pydantic import BaseModel, Field

# 1. 业务逻辑：调用外部搜索引擎 API
def web_medical_search(query: str) -> str:
    """使用搜索引擎获取外部医学知识"""
    print(f"🌐 [网络检索] 正在全网搜索: {query}")
    
    # 这里以免费的 DuckDuckGo API 为例（生产环境建议换成 SerpAPI 或 Google Custom Search）
    # 注意：DuckDuckGo 的简易 API 有时不稳定，仅作结构演示
    try:
        url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode('utf-8'))
            abstract = data.get("Abstract", "")
            
            if abstract:
                return f"网络检索结果:\n{abstract}"
            else:
                return "网络检索未返回直接的医学摘要，可能需要更换更准确的医学术语进行检索。"
    except Exception as e:
        return f"网络检索模块出错: {str(e)}"

# 2. 参数校验模型
class WebSearchArgs(BaseModel):
    query: str = Field(..., description="用于搜索引擎的查询语句，可以包含更泛化的搜索词")

# 3. 技能契约
SKILL_DEF = {
    "name": "web_medical_search",
    "description": "仅当本地检索 (search_local_papers) 失败或返回无数据时，才使用此工具进行互联网医学信息检索。",
    "func": web_medical_search,
    "args_schema": WebSearchArgs
}