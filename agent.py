import json
from openai import OpenAI
from registry import SkillRegistry, start_watchdog

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

GLOBAL_REGISTRY = SkillRegistry()
start_watchdog(GLOBAL_REGISTRY, "skills")

SYSTEM_PROMPT = """
你是医学信息整理助手。
规则：
1. 只能基于 evidence_json 作答，不要编造。
2. 如果 evidence_json.source == "pubmed"，说明网络检索已经完成，不要再说“我将去查询网络”。
3. 输出分四段：
   - 结论
   - 依据
   - 来源
   - 不确定性与就医提醒
4. 这是医学信息整理，不是个体化诊断或处方。
5. 当 evidence_json.enough == false 时，要明确说明证据不足，不要装作已经查到了明确结论。
"""

def call_medical_retrieve(user_prompt: str) -> dict:
    tool_info = GLOBAL_REGISTRY.get_skill_info("medical_retrieve")
    if not tool_info:
        raise RuntimeError("未找到 medical_retrieve。请确认 skills/medical_retrieve.py 已加载。")

    validated_args = tool_info["args_schema"](query=user_prompt)
    result = tool_info["func"](**validated_args.model_dump())

    if not isinstance(result, dict):
        raise RuntimeError("medical_retrieve 必须返回 dict/JSON 结构。")

    return result

def answer_with_evidence(user_prompt: str, evidence: dict) -> str:
    response = client.chat.completions.create(
        model="qwen2.5:7b",
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"用户问题：{user_prompt}\n\n"
                    f"evidence_json:\n{json.dumps(evidence, ensure_ascii=False, indent=2)}\n\n"
                    "请严格基于 evidence_json 回答。"
                ),
            },
        ],
    )
    return response.choices[0].message.content or "未生成回答。"

def run_agent_daemon():
    print("Medical Agent 已就绪。输入 'exit' 退出。")

    while True:
        try:
            user_prompt = input("\n[User]> ").strip()
            if user_prompt == "exit":
                break
            if not user_prompt:
                continue

            print("\n⏳ [思考与执行] 正在调度技能: medical_retrieve ...")
            evidence = call_medical_retrieve(user_prompt)

            preview = json.dumps(evidence, ensure_ascii=False)
            print(f"🔧 [执行结果] {preview[:320]}...")

            if evidence.get("queries_tried"):
                print(f"🔎 [PubMed queries] {evidence['queries_tried']}")

            if evidence.get("debug", {}).get("errors"):
                print(f"🪵 [调试信息] {evidence['debug']['errors']}")

            print("🧠 [汇总] 工具数据已获取，正在进行最终信息整合...\n")

            answer = answer_with_evidence(user_prompt, evidence)
            print(f"[Agent]> {answer}")

        except KeyboardInterrupt:
            print("\n系统正在安全关闭...")
            break
        except Exception as e:
            print(f"[Agent]> 系统错误: {e}")

if __name__ == "__main__":
    run_agent_daemon()
