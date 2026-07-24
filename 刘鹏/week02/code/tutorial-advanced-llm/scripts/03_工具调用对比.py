"""
03_工具调用对比.py

==================== 功能说明 ====================

对比使用工具（Tool Calls / Function Calling）和不使用工具时的差异：
    1. 响应结构差异（finish_reason、tool_calls 字段）
    2. 消息历史拼接差异（role="tool" 消息）
    3. 多轮工具调用循环（Agent Loop）
    4. 并行工具调用
    5. tool_choice 控制策略

核心知识点：
    - AI 不能执行工具，只是"说"要调什么，真正执行的是你的代码
    - tool_calls 中的 arguments 是 JSON 字符串，需要 json.loads() 解析
    - 喂回结果时 tool_call_id 必须一一对应
    - finish_reason="tool_calls" 表示模型要调工具（还没回答完）
    - finish_reason="stop" 表示模型回答完毕

运行方式：
    cd tutorial-advanced-llm
    python scripts/03_工具调用对比.py

==================================================
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from advanced_llm import create_client, cfg, create_default_registry
from advanced_llm.models import ChatResult


def print_separator(title: str = ""):
    if title:
        print(f"\n{'='*65}")
        print(f"  {title}")
        print(f"{'='*65}")
    else:
        print(f"\n{'-'*65}")


# ═══════════════════════════════════════════════════════════
# 1. 不使用工具 vs 使用工具 — 响应结构对比
# ═══════════════════════════════════════════════════════════

def demo_with_vs_without_tools():
    """
    对比同一问题在"有工具"和"无工具"时的响应差异

    关键差异：
        - 无工具：模型直接用已有知识回答（可能编造/幻觉）
        - 有工具：模型判断需要外部信息 → 返回 tool_calls → 你执行 → 喂回结果 → 模型最终回答

    响应结构对比：
        无工具：finish_reason="stop", message.content="回答文本"
        有工具：finish_reason="tool_calls", message.tool_calls=[...], message.content=None
    """
    print_separator("1. 不使用工具 vs 使用工具 — 响应结构对比")

    client = create_client()
    registry = create_default_registry()
    tools = registry.get_openai_schemas()

    question = "北京今天天气怎么样？"
    messages = [
        {"role": "system", "content": "你是智能助手。当前日期：2026-07-22。"},
        {"role": "user", "content": question},
    ]

    # --- 不使用工具 ---
    print(f"\n  问题: {question}")
    print(f"\n  【不使用工具】")
    result_no_tool = client.chat(messages)
    print(f"    finish_reason: {result_no_tool.finish_reason}")
    print(f"    content: {result_no_tool.content}")
    print(f"    tool_calls: {result_no_tool.tool_calls}")
    print(f"    → 模型只能用训练数据中的知识回答（可能过时/编造）")

    # --- 使用工具 ---
    print(f"\n  【使用工具】")
    result_with_tool = client.chat(messages, tools=tools)
    print(f"    finish_reason: {result_with_tool.finish_reason}")
    print(f"    content: {result_with_tool.content}")
    print(f"    has_tool_calls: {result_with_tool.has_tool_calls}")

    if result_with_tool.has_tool_calls:
        for tc in result_with_tool.tool_calls:
            print(f"    tool_call: id={tc.id}, name={tc.name}, args={tc.arguments}")
        print(f"    → 模型判断需要调用外部工具获取实时信息")

    print(f"""
    【响应结构对比总结】
    ┌──────────────────┬─────────────────────────┬─────────────────────────────┐
    │ 字段             │ 不使用工具              │ 使用工具（模型要调用时）      │
    ├──────────────────┼─────────────────────────┼─────────────────────────────┤
    │ finish_reason    │ "stop"                  │ "tool_calls"                │
    │ message.content  │ 回答文本                │ None 或空                   │
    │ message.tool_calls│ None                   │ [{id, function:{name,args}}]│
    │ 后续操作         │ 直接展示给用户          │ 执行工具 → 喂回结果 → 再调API│
    └──────────────────┴─────────────────────────┴─────────────────────────────┘
    """)


# ═══════════════════════════════════════════════════════════
# 2. 完整工具调用流程（单次）
# ═══════════════════════════════════════════════════════════

def demo_single_tool_call_flow():
    """
    完整的单次工具调用流程

    流程：
        1. 用户提问 + 工具清单 → 发给模型
        2. 模型返回 tool_calls（要调哪个工具、什么参数）
        3. 你的代码执行工具，拿到结果
        4. 把结果作为 role="tool" 消息喂回模型
        5. 模型根据工具结果生成最终自然语言回答
    """
    print_separator("2. 完整工具调用流程（单次）")

    client = create_client()
    registry = create_default_registry()
    tools = registry.get_openai_schemas()

    messages = [
        {"role": "system", "content": "你是智能客服助手。当前日期：2026-07-22。"},
        {"role": "user", "content": "北京今天天气怎么样？"},
    ]

    print(f"  用户: {messages[-1]['content']}")

    # 第一次调用：模型决定是否使用工具
    print(f"\n  [Step 1] 发送请求（带工具清单）...")
    result = client.chat(messages, tools=tools)

    if not result.has_tool_calls:
        print(f"  模型直接回答: {result.content}")
        return

    # 模型要调工具
    print(f"  [Step 2] 模型请求调用工具:")
    for tc in result.tool_calls:
        print(f"    → {tc.name}({tc.arguments})")

    # 执行工具
    print(f"\n  [Step 3] 执行工具...")
    # 关键：把 assistant 的 tool_calls 消息加入历史
    messages.append({
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
    })

    for tc in result.tool_calls:
        tool_result = registry.execute_tool_call(tc)
        print(f"    ← {tc.name} 结果: {tool_result.result}")

        # 关键：把工具结果作为 role="tool" 消息加入历史
        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,  # 必须与 tool_call.id 对应！
            "content": tool_result.result,
        })

    # 第二次调用：模型根据工具结果生成最终回答
    print(f"\n  [Step 4] 将工具结果喂回模型...")
    final_result = client.chat(messages, tools=tools)
    print(f"  [Step 5] 最终回答: {final_result.content}")

    print(f"""
    【messages 历史变化过程】
    初始: [system, user]
    第一次调用后: [system, user, assistant(tool_calls)]
    执行工具后: [system, user, assistant(tool_calls), tool(结果)]
    第二次调用后: 得到最终自然语言回答
    """)


# ═══════════════════════════════════════════════════════════
# 3. 多轮工具调用循环（Agent Loop）
# ═══════════════════════════════════════════════════════════

def demo_agent_loop():
    """
    多轮工具调用循环（Agent Loop）

    一个复杂问题可能需要多次工具调用：
        用户: "北京今天气温多少？最高温和最低温的平均值是多少？"
        → 模型先调 get_weather 获取温度
        → 再调 calculate 计算平均值
        → 最终回答

    生产级要点：
        - 必须设 max_iterations 防止死循环
        - 每次循环检查 finish_reason
        - 工具执行失败要返回结构化错误给模型
    """
    print_separator("3. 多轮工具调用循环（Agent Loop）")

    client = create_client()
    registry = create_default_registry()
    tools = registry.get_openai_schemas()

    messages = [
        {"role": "system", "content": "你是智能助手，可使用工具回答问题。当前日期：2026-07-22。"},
        {"role": "user", "content": "帮我算一下 2 的 10 次方等于多少？然后查一下北京的天气。"},
    ]

    print(f"  用户: {messages[-1]['content']}\n")

    max_iterations = 5
    for iteration in range(1, max_iterations + 1):
        print(f"  --- 第 {iteration} 轮 ---")

        result = client.chat(messages, tools=tools)

        if not result.has_tool_calls:
            # 模型给出最终回答
            print(f"  ✅ 最终回答: {result.content}")
            break

        # 模型要调工具
        print(f"  模型请求 {len(result.tool_calls)} 个工具调用:")

        # 把 assistant 消息加入历史
        messages.append({
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
        })

        # 执行所有工具调用
        for tc in result.tool_calls:
            tool_result = registry.execute_tool_call(tc)
            print(f"    → {tc.name}({tc.arguments})")
            print(f"    ← {tool_result.result}")

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": tool_result.result,
            })

        print()
    else:
        print(f"  ⚠️  达到最大迭代次数 {max_iterations}，未收敛！")


# ═══════════════════════════════════════════════════════════
# 4. tool_choice 控制策略对比
# ═══════════════════════════════════════════════════════════

def demo_tool_choice_strategies():
    """
    tool_choice 参数控制模型使用工具的行为

    可选值：
        - "auto": 模型自行决定（默认）
        - "none": 禁止使用工具（即使传了 tools 也不调）
        - "required": 必须使用至少一个工具
        - {"type":"function","function":{"name":"xxx"}}: 强制调用指定工具
    """
    print_separator("4. tool_choice 控制策略对比")

    client = create_client()
    registry = create_default_registry()
    tools = registry.get_openai_schemas()

    question = "北京今天天气怎么样？"
    base_messages = [
        {"role": "system", "content": "你是智能助手。"},
        {"role": "user", "content": question},
    ]

    # --- auto（默认）---
    print(f"\n  问题: {question}")
    print(f"\n  [tool_choice='auto'] 模型自行决定:")
    result = client.chat(base_messages, tools=tools, tool_choice="auto")
    if result.has_tool_calls:
        print(f"    → 决定调用工具: {result.tool_calls[0].name}")
    else:
        print(f"    → 直接回答: {result.content[:60]}")

    # --- none（禁止）---
    print(f"\n  [tool_choice='none'] 禁止使用工具:")
    result = client.chat(base_messages, tools=tools, tool_choice="none")
    print(f"    → 直接回答: {result.content[:80]}")
    print(f"    （即使有天气工具也不调用，只能用已有知识回答）")

    # --- 强制指定工具 ---
    print(f"\n  [tool_choice=指定calculate] 强制调用计算工具:")
    calc_messages = [
        {"role": "system", "content": "你是智能助手。"},
        {"role": "user", "content": "123乘以456等于多少？"},
    ]
    result = client.chat(
        calc_messages,
        tools=tools,
        tool_choice='{"type":"function","function":{"name":"calculate"}}',
    )
    if result.has_tool_calls:
        print(f"    → 强制调用: {result.tool_calls[0].name}({result.tool_calls[0].arguments})")
    else:
        print(f"    → 回答: {result.content[:60]}")

    print(f"""
    【tool_choice 使用场景】
    - auto: 大多数场景的默认选择，让模型智能判断
    - none: 明确不需要工具时（如纯聊天、翻译），避免误触发
    - required: 确保模型一定使用工具（如必须查询数据库才能回答）
    - 指定工具: 明确知道该用哪个工具时（如用户明确说"帮我计算"）
    """)


# ═══════════════════════════════════════════════════════════
# 5. 工具调用中的错误处理
# ═══════════════════════════════════════════════════════════

def demo_tool_error_handling():
    """
    工具执行失败时的优雅处理

    生产级要点：
        - 工具执行失败不能直接抛异常给用户
        - 返回结构化错误给模型，让模型决定如何降级
        - 模型会根据错误信息生成友好的用户提示
    """
    print_separator("5. 工具调用中的错误处理")

    client = create_client()
    registry = create_default_registry()
    tools = registry.get_openai_schemas()

    # 模拟一个会失败的调用（查询不存在的城市）
    messages = [
        {"role": "system", "content": "你是智能助手。如果工具返回错误，请友好地告知用户。"},
        {"role": "user", "content": "查一下火星的天气"},
    ]

    print(f"  用户: {messages[-1]['content']}\n")

    result = client.chat(messages, tools=tools)

    if result.has_tool_calls:
        messages.append({
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
        })

        for tc in result.tool_calls:
            tool_result = registry.execute_tool_call(tc)
            print(f"  工具调用: {tc.name}({tc.arguments})")
            print(f"  执行结果: {tool_result.result}")
            print(f"  是否成功: {tool_result.success}")

            # 即使失败也喂回结果（让模型知道失败了）
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": tool_result.result,  # 包含错误信息
            })

        # 模型根据错误信息生成友好回答
        final = client.chat(messages, tools=tools)
        print(f"\n  模型最终回答: {final.content}")
        print(f"  → 模型看到工具返回'暂未收录'，自动降级为友好提示")


# ═══════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "╔" + "═"*63 + "╗")
    print("║   03_工具调用对比 — 使用工具 vs 不使用工具的差异            ║")
    print("╚" + "═"*63 + "╝")

    # 1. 响应结构对比
    demo_with_vs_without_tools()

    # 2. 完整工具调用流程
    demo_single_tool_call_flow()

    # 3. Agent Loop
    demo_agent_loop()

    # 4. tool_choice 策略
    demo_tool_choice_strategies()

    # 5. 错误处理
    demo_tool_error_handling()

    print_separator("演示完成")
