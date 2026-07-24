"""
04_思考+工具+多轮交互.py

==================== 功能说明 ====================

本脚本演示思考模式（Thinking）+ 工具调用（Tool Calls）+ 多轮对话三者结合时的
关键规则与最佳实践。

核心规则（DeepSeek 官方文档明确要求）：
    ┌─────────────────────────────────────────────────────────────────────┐
    │ 无工具调用时：assistant 的 reasoning_content 无需回传               │
    │              → 多轮历史中只保留 content 即可（节省 token）          │
    │                                                                     │
    │ 有工具调用时：assistant 的 reasoning_content 必须回传               │
    │              → 后续所有轮次的 messages 中必须包含完整 reasoning      │
    │              → 否则会导致上下文不连贯或 API 报错                    │
    └─────────────────────────────────────────────────────────────────────┘

本脚本对比演示：
    1. 思考+工具调用时 reasoning 回传 vs 不回传的差异
    2. 多轮对话中正确拼接 reasoning 的方式
    3. 生产级 Agent 循环（思考+工具+多轮）

运行方式：
    cd tutorial-advanced-llm
    python scripts/04_思考+工具+多轮交互.py

==================================================
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from advanced_llm import create_client, cfg, create_default_registry
from advanced_llm.context_manager import ContextManager, HybridStrategy


def print_separator(title: str = ""):
    if title:
        print(f"\n{'='*65}")
        print(f"  {title}")
        print(f"{'='*65}")
    else:
        print(f"\n{'-'*65}")


# ═══════════════════════════════════════════════════════════
# 1. 核心规则演示：reasoning_content 回传规则
# ═══════════════════════════════════════════════════════════

def demo_reasoning_feedback_rule():
    """
    演示 reasoning_content 在多轮对话中的回传规则

    规则总结：
        场景A（无工具调用）：
            第1轮: user → assistant(content + reasoning)
            第2轮: messages = [system, user1, assistant1(只保留content), user2]
            → reasoning 可以丢弃，节省 token

        场景B（有工具调用）：
            第1轮: user → assistant(reasoning + tool_calls) → tool(结果) → assistant(最终回答)
            第2轮: messages 中必须包含 assistant 的 reasoning_content
            → reasoning 必须保留，否则 API 报错或上下文断裂
    """
    print_separator("1. reasoning_content 回传规则演示")

    client = create_client()
    model_cfg = cfg.active_model

    if not model_cfg.supports_thinking:
        print(f"  ⚠️  当前模型 [{model_cfg.name}] 不支持思考模式")
        print(f"      此演示需要 DeepSeek 或 Claude 模型")
        print(f"      以下展示代码逻辑（伪代码）：\n")
        _print_reasoning_rule_pseudocode()
        return

    # --- 场景A：无工具调用，reasoning 可丢弃 ---
    print("\n  【场景A：无工具调用 — reasoning 可丢弃】\n")

    messages_a = [
        {"role": "system", "content": "你是数学老师。"},
        {"role": "user", "content": "1+1等于？"},
    ]

    result_a = client.chat(messages_a, enable_thinking=True)
    print(f"  第1轮回答: {result_a.content}")
    print(f"  有思考内容: {bool(result_a.reasoning)}")

    # 正确做法：只保留 content，丢弃 reasoning
    messages_a.append({"role": "assistant", "content": result_a.content})
    messages_a.append({"role": "user", "content": "那2+2呢？"})

    result_a2 = client.chat(messages_a, enable_thinking=True)
    print(f"  第2轮回答: {result_a2.content}")
    print(f"  ✅ 无工具调用时，丢弃 reasoning 后对话正常\n")

    # --- 场景B：有工具调用，reasoning 必须保留 ---
    print("  【场景B：有工具调用 — reasoning 必须保留】\n")

    registry = create_default_registry()
    tools = registry.get_openai_schemas()

    messages_b = [
        {"role": "system", "content": "你是智能助手。当前日期：2026-07-22。"},
        {"role": "user", "content": "北京今天天气怎么样？"},
    ]

    result_b = client.chat(messages_b, tools=tools, enable_thinking=True)

    if result_b.has_tool_calls:
        print(f"  模型请求工具: {result_b.tool_calls[0].name}")
        print(f"  有思考内容: {bool(result_b.reasoning)}")

        # 正确做法：保留 reasoning + tool_calls
        assistant_msg = {
            "role": "assistant",
            "content": result_b.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments},
                }
                for tc in result_b.tool_calls
            ],
        }
        # 关键：有工具调用时必须包含 reasoning
        if result_b.reasoning:
            assistant_msg["reasoning_content"] = result_b.reasoning

        messages_b.append(assistant_msg)

        # 执行工具
        for tc in result_b.tool_calls:
            tool_result = registry.execute_tool_call(tc)
            messages_b.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": tool_result.result,
            })
            print(f"  工具结果: {tool_result.result}")

        # 获取最终回答
        final = client.chat(messages_b, tools=tools, enable_thinking=True)
        print(f"  最终回答: {final.content}")

        # 继续第二轮对话
        # 关键：历史中的 reasoning_content 必须保留
        final_msg = {"role": "assistant", "content": final.content}
        if final.reasoning:
            final_msg["reasoning_content"] = final.reasoning
        messages_b.append(final_msg)
        messages_b.append({"role": "user", "content": "那上海呢？"})

        result_b2 = client.chat(messages_b, tools=tools, enable_thinking=True)
        print(f"\n  第2轮回答: {result_b2.content}")
        print(f"  ✅ 有工具调用时，保留 reasoning 后多轮对话正常")


def _print_reasoning_rule_pseudocode():
    """打印 reasoning 回传规则的伪代码"""
    print("""
    # ═══ 场景A：无工具调用 ═══
    # reasoning 可以安全丢弃

    result = client.chat(messages, enable_thinking=True)
    # ✅ 正确：只保留 content
    messages.append({"role": "assistant", "content": result.content})
    # ❌ 错误（浪费token）：
    # messages.append({"role": "assistant", "content": result.content,
    #                  "reasoning_content": result.reasoning})


    # ═══ 场景B：有工具调用 ═══
    # reasoning 必须保留！

    result = client.chat(messages, tools=tools, enable_thinking=True)
    if result.has_tool_calls:
        assistant_msg = {
            "role": "assistant",
            "content": result.content,
            "tool_calls": [...],
        }
        # ✅ 必须包含 reasoning（否则后续轮次报错）
        if result.reasoning:
            assistant_msg["reasoning_content"] = result.reasoning
        messages.append(assistant_msg)

        # 执行工具并喂回结果
        for tc in result.tool_calls:
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": ...})
    """)


# ═══════════════════════════════════════════════════════════
# 2. 对比：回传 vs 不回传 reasoning 的差异
# ═══════════════════════════════════════════════════════════

def demo_reasoning_comparison():
    """
    对比在多轮对话中回传/不回传 reasoning 的差异

    观察点：
        - Token 消耗差异（reasoning 可能很长，占大量 token）
        - 回答连贯性差异（有工具调用时不回传可能导致断裂）
        - 成本差异
    """
    print_separator("2. 回传 vs 不回传 reasoning 对比")

    client = create_client()
    model_cfg = cfg.active_model

    if not model_cfg.supports_thinking:
        print(f"  ⚠️  当前模型不支持思考模式，展示理论对比：\n")
        _print_comparison_table()
        return

    question = "9.11 和 9.8 哪个大？"

    # --- 方式1：不回传 reasoning ---
    print(f"  问题: {question}\n")
    print("  【方式1：不回传 reasoning（节省 token）】")

    messages_1 = [
        {"role": "system", "content": "你是数学老师。"},
        {"role": "user", "content": question},
    ]
    r1 = client.chat(messages_1, enable_thinking=True)
    print(f"  第1轮回答: {r1.content}")
    print(f"  第1轮 reasoning 长度: {len(r1.reasoning or '')} 字符")

    # 只保留 content
    messages_1.append({"role": "assistant", "content": r1.content})
    messages_1.append({"role": "user", "content": "为什么？请详细解释。"})
    r1_2 = client.chat(messages_1, enable_thinking=True)
    print(f"  第2轮回答: {r1_2.content[:80]}...")
    print(f"  第2轮 prompt_tokens: {r1_2.usage.get('prompt_tokens', '?') if r1_2.usage else '?'}")

    # --- 方式2：回传 reasoning ---
    print(f"\n  【方式2：回传 reasoning（完整上下文）】")

    messages_2 = [
        {"role": "system", "content": "你是数学老师。"},
        {"role": "user", "content": question},
    ]
    r2 = client.chat(messages_2, enable_thinking=True)

    # 保留 reasoning
    assistant_msg = {"role": "assistant", "content": r2.content}
    if r2.reasoning:
        assistant_msg["reasoning_content"] = r2.reasoning
    messages_2.append(assistant_msg)
    messages_2.append({"role": "user", "content": "为什么？请详细解释。"})
    r2_2 = client.chat(messages_2, enable_thinking=True)
    print(f"  第2轮回答: {r2_2.content[:80]}...")
    print(f"  第2轮 prompt_tokens: {r2_2.usage.get('prompt_tokens', '?') if r2_2.usage else '?'}")

    # 对比
    if r1_2.usage and r2_2.usage:
        t1 = r1_2.usage.get("prompt_tokens", 0)
        t2 = r2_2.usage.get("prompt_tokens", 0)
        print(f"\n  【Token 消耗对比】")
        print(f"    不回传 reasoning: {t1} prompt_tokens")
        print(f"    回传 reasoning:   {t2} prompt_tokens")
        print(f"    差异: +{t2 - t1} tokens (回传多消耗 {((t2-t1)/max(t1,1)*100):.0f}%)")

    _print_comparison_table()


def _print_comparison_table():
    """打印对比总结表"""
    print("""
    【reasoning 回传策略总结】
    ┌──────────────────┬───────────────────────┬────────────────────────────┐
    │ 场景             │ 是否回传 reasoning    │ 原因                       │
    ├──────────────────┼───────────────────────┼────────────────────────────┤
    │ 无工具调用       │ ❌ 不回传（推荐）     │ API 会忽略，浪费 token     │
    │ 有工具调用       │ ✅ 必须回传           │ 否则上下文断裂/API报错     │
    │ 需要调试/审计    │ ✅ 回传（可选）       │ 保留完整推理链路           │
    │ 成本敏感         │ ❌ 不回传             │ 节省 30%~70% prompt token  │
    └──────────────────┴───────────────────────┴────────────────────────────┘

    【实践建议】
    - 普通多轮对话：只存 content，reasoning 存日志（不传 API）
    - Agent/工具调用场景：完整保留 reasoning + tool_calls + tool results
    - 生产环境：reasoning 单独存储（对象存储/日志），主表只存 content
    """)


# ═══════════════════════════════════════════════════════════
# 3. 生产级 Agent 循环（思考+工具+多轮）
# ═══════════════════════════════════════════════════════════

def demo_production_agent_loop():
    """
    生产级 Agent 循环：思考 + 工具调用 + 多轮对话

    这是最复杂的场景，结合了所有能力：
        1. 模型先思考（reasoning）
        2. 决定调用工具（tool_calls）
        3. 执行工具，喂回结果
        4. 模型可能继续调工具或给出最终回答
        5. 用户继续追问（多轮）
        6. 历史中必须保留 reasoning（因为有工具调用）

    生产级要点：
        - max_iterations 防死循环
        - 工具执行超时控制
        - 错误结构化返回
        - reasoning 完整保留
        - token 消耗监控
    """
    print_separator("3. 生产级 Agent 循环（思考+工具+多轮）")

    client = create_client()
    registry = create_default_registry()
    tools = registry.get_openai_schemas()
    model_cfg = cfg.active_model

    enable_thinking = model_cfg.supports_thinking
    print(f"  模型: {model_cfg.name} | 思考模式: {enable_thinking}\n")

    # 使用 ContextManager 管理历史
    cm = ContextManager(
        system_prompt="你是智能助手，可使用工具回答问题。当前日期：2026-07-22。",
        strategy=HybridStrategy(max_rounds=10, max_tokens=8000),
    )

    # 模拟多轮对话
    user_questions = [
        "北京今天天气怎么样？",
        "那帮我算一下 28 加 35 等于多少？（就是刚才北京最高温和最低温的和）",
    ]

    for q_idx, question in enumerate(user_questions, 1):
        print(f"\n  {'─'*50}")
        print(f"  [第 {q_idx} 轮] 用户: {question}")
        print(f"  {'─'*50}")

        cm.add_user_message(question)

        # Agent Loop
        max_iterations = 5
        for iteration in range(1, max_iterations + 1):
            messages = cm.get_messages()
            result = client.chat(
                messages,
                tools=tools,
                enable_thinking=enable_thinking,
            )

            if result.reasoning:
                print(f"  💭 思考: {result.reasoning[:80]}...")

            if not result.has_tool_calls:
                # 最终回答
                print(f"  🤖 回答: {result.content}")

                # 添加到历史（有工具调用历史时保留 reasoning）
                cm.add_assistant_message(
                    content=result.content,
                    reasoning=result.reasoning,
                    include_reasoning=enable_thinking,  # 有工具调用的轮次需要保留
                )
                break

            # 工具调用
            print(f"  🔧 工具调用 (第{iteration}次):")

            # 构建 assistant 消息（含 tool_calls + reasoning）
            assistant_msg = {
                "role": "assistant",
                "content": result.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": tc.arguments},
                    }
                    for tc in result.tool_calls
                ],
            }
            # 关键：有工具调用时 reasoning 必须保留
            if result.reasoning:
                assistant_msg["reasoning_content"] = result.reasoning

            # 直接操作内部 messages（因为 ContextManager 不直接支持 tool 消息的复杂拼接）
            cm._messages.append(assistant_msg)

            # 执行工具
            for tc in result.tool_calls:
                tool_result = registry.execute_tool_call(tc)
                print(f"    → {tc.name}({tc.arguments})")
                print(f"    ← {tool_result.result}")

                cm.add_tool_result(
                    tool_call_id=tc.id,
                    content=tool_result.result,
                )
        else:
            print(f"  ⚠️  达到最大迭代次数！")

    print(f"\n  最终历史: {cm.message_count} 条消息, 估算 {cm.get_token_estimate()} tokens")


# ═══════════════════════════════════════════════════════════
# 4. 思考内容对后续对话的影响
# ═══════════════════════════════════════════════════════════

def demo_thinking_impact_on_followup():
    """
    对比：将思考内容喂给后续对话 vs 不喂的差异

    观察：
        - 喂回 reasoning：模型能"回忆"之前的推理过程，回答更连贯
        - 不喂 reasoning：模型只看到最终答案，可能丢失推理上下文
    """
    print_separator("4. 思考内容对后续对话的影响")

    client = create_client()
    model_cfg = cfg.active_model

    if not model_cfg.supports_thinking:
        print(f"  ⚠️  当前模型不支持思考模式，展示理论说明\n")
        print("""
    【理论说明】

    当模型在第1轮进行了复杂推理后：
    - 回传 reasoning：第2轮模型能看到完整推理链，追问时更精准
    - 不回传 reasoning：第2轮模型只看到结论，追问细节时可能重新推理

    示例：
    第1轮: "证明根号2是无理数" → reasoning 包含完整反证法过程
    第2轮: "刚才证明中为什么假设p/q是最简分数？"

    回传 reasoning → 模型能直接引用之前的推理步骤回答
    不回传 reasoning → 模型需要重新理解上下文，可能回答不够精准
        """)
        return

    # 第1轮：复杂推理
    q1 = "用反证法证明：根号2是无理数。"
    print(f"  第1轮问题: {q1}\n")

    messages = [
        {"role": "system", "content": "你是数学教授，回答严谨。"},
        {"role": "user", "content": q1},
    ]

    r1 = client.chat(messages, enable_thinking=True, max_tokens=2000)
    print(f"  回答: {r1.content[:100]}...")
    print(f"  reasoning 长度: {len(r1.reasoning or '')} 字符")

    # 第2轮追问
    q2 = "你刚才的证明中，为什么要假设 p/q 是最简分数？如果不用这个假设会怎样？"
    print(f"\n  第2轮追问: {q2}\n")

    # 方式A：不回传 reasoning
    messages_a = messages + [
        {"role": "assistant", "content": r1.content},
        {"role": "user", "content": q2},
    ]
    ra = client.chat(messages_a, enable_thinking=True, max_tokens=1000)
    print(f"  【不回传reasoning】: {ra.content[:100]}...")

    # 方式B：回传 reasoning
    assistant_with_reasoning = {"role": "assistant", "content": r1.content}
    if r1.reasoning:
        assistant_with_reasoning["reasoning_content"] = r1.reasoning
    messages_b = messages + [
        assistant_with_reasoning,
        {"role": "user", "content": q2},
    ]
    rb = client.chat(messages_b, enable_thinking=True, max_tokens=1000)
    print(f"  【回传reasoning】: {rb.content[:100]}...")

    if ra.usage and rb.usage:
        print(f"\n  Token对比: 不回传={ra.usage.get('prompt_tokens',0)}, "
              f"回传={rb.usage.get('prompt_tokens',0)}")


# ═══════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "╔" + "═"*63 + "╗")
    print("║   04_思考+工具+多轮交互 — reasoning 回传规则与最佳实践      ║")
    print("╚" + "═"*63 + "╝")

    # 1. reasoning 回传规则
    demo_reasoning_feedback_rule()

    # 2. 回传 vs 不回传对比
    demo_reasoning_comparison()

    # 3. 生产级 Agent 循环
    demo_production_agent_loop()

    # 4. 思考内容对后续对话的影响（取消注释运行）
    # demo_thinking_impact_on_followup()

    print_separator("演示完成")
