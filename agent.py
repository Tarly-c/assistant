import os
import json
import time
from openai import OpenAI
from registry import SkillRegistry, start_watchdog

client = OpenAI(
    base_url="http://localhost:11434/v1", # Ollama 默认的本地服务地址 + /v1 路径
    api_key="ollama" # 必须传一个非空字符串，写什么都可以，通常写 "ollama"
)

# 初始化全局并发安全的注册表
GLOBAL_REGISTRY = SkillRegistry()

# 启动后台监听线程
start_watchdog(GLOBAL_REGISTRY, "skills")

def run_agent_daemon():
    """守护进程"""
    print("🚀 Agent 守护进程已就绪。输入指令进行对话，输入 'exit' 退出。")
    
    messages = [{"role": "system", "content": "你是一个严谨的专业助理。请按需调用工具，并根据工具返回的结果为用户提供最终解答。"}]

    # 1. 外层循环：生命周期与用户交互
    while True:
        try:
            user_prompt = input("\n[User]> ")
            if user_prompt.strip() == 'exit':
                break
                
            messages.append({"role": "user", "content": user_prompt})
            
            # 2. 内层循环：Agent 内部的“推理-行动”闭环 (ReAct Loop)
            step_count = 0
            max_steps = 5 # 强制设定最大步数，防止模型陷入无休止的死循环报错中
            
            while step_count < max_steps:
                step_count += 1
                
                # 每次推理前，实时获取最新的技能清单
                current_tools = GLOBAL_REGISTRY.get_all_llm_schemas()
                
                if not current_tools:
                    print("[Agent]> ⚠️ 警告：当前系统未挂载任何技能。")
                    # 如果没有工具，也要进行一次普通回复并退出内层循环
                    response = client.chat.completions.create(
                        model="qwen2.5:7b",
                        messages=messages,
                        temperature=0.1
                    )
                    message = response.choices[0].message
                    messages.append(message)
                    print(f"[Agent]> {message.content}")
                    break

                response = client.chat.completions.create(
                    model="qwen2.5:7b",
                    messages=messages,
                    tools=current_tools,
                    temperature=0.1
                )
                
                message = response.choices[0].message
                messages.append(message)
                
                # ==========================================
                # 退出条件：大模型没有发出工具调用指令，说明得出了最终答案
                # ==========================================
                if not message.tool_calls:
                    print(f"\n[Agent]> {message.content}")
                    break # 跳出内层循环，等待用户的下一个问题

                # ==========================================
                # 执行条件：大模型请求调用工具
                # ==========================================
                for tool_call in message.tool_calls:
                    func_name = tool_call.function.name
                    raw_args = tool_call.function.arguments
                    
                    print(f"\n⏳ [思考与执行] 正在调度技能: {func_name} ...")
                    
                    # 从注册表安全地取回函数指针
                    tool_info = GLOBAL_REGISTRY.get_skill_info(func_name)
                    
                    if not tool_info:
                        tool_result = f"System Error: 技能 {func_name} 已被物理移除或未加载。"
                    else:
                        try:
                            parsed_args = json.loads(raw_args)
                            validated_args = tool_info["args_schema"](**parsed_args)
                            # 模拟耗时操作，给予用户执行反馈
                            time.sleep(0.5) 
                            # 执行真实的业务代码
                            tool_result = tool_info["func"](**validated_args.model_dump())
                        except Exception as e:
                            tool_result = f"Runtime Error: {str(e)}"
                    
                    # 打印内部执行状态，对开发者透明
                    print(f"🔧 [执行结果] {str(tool_result)[:100]}...") 
                    
                    # 将执行结果塞回上下文
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": func_name,
                        "content": str(tool_result)
                    })
                
                print("🧠 [汇总] 工具数据已获取，正在进行最终信息整合...\n")
                # 此时内层 while 循环继续，带着挂载了 tool 结果的 messages 再次请求大模型

            if step_count >= max_steps:
                print("\n[Agent]> ⚠️ 任务超出最大推理步数，已强制中断，请简化您的请求。")

        except KeyboardInterrupt:
            print("\n系统正在安全关闭...")
            break

if __name__ == "__main__":
    run_agent_daemon()