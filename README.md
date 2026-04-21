#### 环境搭建

模型选择：ollama本地部署的Qwen2.5:7b与nomic-embed-text

数据集选择：https://medlineplus.gov/xml.html 

- 选择MedlinePlus Health Topic XML文件，包含了广泛的健康话题，summary中包含了定义，症状，治疗方法，当然还有很多其他的东西。
- 使用脚本清洗数据，保留summary关键信息，并把topics分条切出，得到markdown文件
- 把markdown分别切成chunk，再使用nomic-embed-text把这样的标准化的数据切片写进chroma，作为语义向量数据库。

#### 工作流：

question-> plan-> local_result-> web_result-> answer

- question：用户提问
- plan：首先将用户提问医学名词标准化，然后给出一种策略：先查本地，是否需要再查外部，外部查什么方向。
- local_result：对查询进行向量化，并在 Chroma 中进行相似度搜索（打分），返回若干个最相关的 chunk，作为回答依据。
- web_result：在本地证据不充分时进行补充。
- answer：基于证据进行整合、压缩和重述，最终生成对用户友好的自然语言答案。

#### 前端

使用cloudflare的tunnelling功能，将本地服务布置在www.mysuperlt.top上。
