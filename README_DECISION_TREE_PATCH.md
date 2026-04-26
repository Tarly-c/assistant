# Decision-tree probe patch for commit21

基于 commit `4aa9b3877fd6cdab8cdf705366eb1d42493d5239`，这组文件把原先手写 `FEATURES` 的病例定位流程替换为：

1. **离线构建一棵问诊决策树**：每个节点对应当前可行病例子集。
2. **节点 probe 自动生成**：从该节点病例的 `description` 中抽取观察片段，使用 TF-IDF 字符 n-gram 向量聚类，并按切分收益选择最有区分度的 probe。
3. **线上沿树问诊**：用户答“是/不是”进入 yes/no 子树；答“不确定”则停留当前节点并触发局部动态 probe 兜底。
4. **没有静态牙疼特征表**：`features.py` 不再维护 `FEATURES = (...)`。
5. **无需 LLM 即可运行**：问题文本先用 `question_seed`，后续可接 LLM 只负责“probe → 自然问句”和“用户回答 → yes/no/uncertain”。

## 覆盖/新增文件

把本目录里的文件复制到仓库根目录，覆盖同名文件：

```text
medical_assistant/config.py
medical_assistant/schemas.py
medical_assistant/graph/state.py
medical_assistant/graph/nodes/final_answer.py
medical_assistant/graph/nodes/normalize.py
medical_assistant/graph/nodes/update_memory.py
medical_assistant/graph/nodes/narrow_cases.py
medical_assistant/graph/nodes/plan_question.py
medical_assistant/graph/nodes/retrieve_cases.py
medical_assistant/services/cases/features.py
medical_assistant/services/cases/memory.py
medical_assistant/services/cases/planner.py
medical_assistant/services/cases/question_tree.py
medical_assistant/services/cases/store.py
scripts/build_question_tree.py
```

## 使用方式

```bash
python scripts/build_question_tree.py --print-tree
python scripts/build_case_index.py --check
python run.py cli --debug
```

默认会把树写到：

```text
resources/case_question_tree.json
```

可通过环境变量改路径：

```bash
export MEDICAL_ASSISTANT_CASE_QUESTION_TREE_FILE=resources/case_question_tree.json
```

## 关键实现点

- `features.py::mine_local_probes()` 是核心算法：从病例子集自动挖 probe。
- `question_tree.py::build_question_tree()` 递归构建树。
- `planner.py::select_question()` 优先使用离线树；树无法切分时启用局部动态 probe；小候选集时进入病例确认问题。
- `memory.py` 保存 probe split 与树节点状态，避免用户答“是”后还问明显互斥的问题。

## 说明

这版没有做强化学习，也没有调用 LLM。它先实现“一个简单决策树”，后续可以在 `PlannedQuestion.text` 生成处接入 LLM 来润色问句，在 `classify_answer()` 处接入 LLM 来理解更复杂的用户回答。
