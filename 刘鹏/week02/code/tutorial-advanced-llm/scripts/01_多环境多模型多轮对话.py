"""
01_多环境多模型多轮对话.py

==================== 功能说明 ====================

演示生产级多环境、多格式（OpenAI/Anthropic）、多模型（DeepSeek/Qwen/Kimi/Claude）
的统一调用与多轮对话管理。

核心知识点：
    1. 工厂模式：create_client() 自动识别 provider 选择客户端
    2. 统一接口：不同格式的模型使用相同的 chat() / chat_stream() 接口
    3. 多轮对话：手动维护 messages 历史（API 无状态）
    4. 环境切换：APP_ENV 控制开发/生产配置隔离

运行方式：
    cd tutorial-advanced-llm
    python scripts/01_多环境多模型多轮对话.py

    # 切换环境
    $env:APP_ENV="prod"; python scripts/01_多环境多模型多轮对话.py

    # 切换模型
    $env:ACTIVE_MODEL="qwen"; python scripts/01_多环境多模型多轮对话.py

==================================================
"""

import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from advanced_llm import create_client, create_all_clients, cfg
from advanced_llm.context_manager import ContextManager, SlidingWindowStrategy


def print_separator(title: str = ""):
    """打印分隔线"""
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
    else:
        print(f"\n{'-'*60}")


# ═══════════════════════════════════════════════════════════
# 1. 环境信息展示
# ═══════════════════════════════════════════════════════════

def demo_env_info():
    """展示当前环境和配置信息"""
    print_separator("1. 当前环境配置")

    print(f"  运行环境: {cfg.APP_ENV}")
    print(f"  激活模型: {cfg.active_model.name}")
    print(f"  模型标识: {cfg.active_model.model}")
    print(f"  API 格式: {cfg.active_model.provider}")
    print(f"  思考模式: {cfg.ENABLE_THINKING}")
    print(f"  可用模型: {cfg.available_models}")
    print(f"  MAX_TOKENS: {cfg.MAX_TOKENS}")
    print(f"  TEMPERATURE: {cfg.TEMPERATURE}")

    print("\n  各模型配置详情：")
    for name in cfg.available_models:
        model_cfg = cfg.get_model(name)
        print(f"    [{name}] provider={model_cfg.provider}, "
              f"model={model_cfg.model}, "
              f"thinking={model_cfg.supports_thinking}")


# ═══════════════════════════════════════════════════════════
# 2. 单模型非流式调用
# ═══════════════════════════════════════════════════════════

def demo_single_call():
    """单模型非流式调用（使用当前激活模型）"""
    print_separator("2. 单模型非流式调用")

    client = create_client()  # 自动使用 ACTIVE_MODEL
    print(f"  客户端: {client}")

    messages = [
        {"role": "system", "content": "你是一个专业的 Python 开发者，回答简洁精准。"},
        {"role": "user", "content": "用一句话解释 Python 的 GIL 是什么？"},
    ]

    result = client.chat(messages)

    print(f"\n  问题: {messages[-1]['content']}")
    print(f"  回答: {result.content}")
    print(f"  Token 消耗: {result.usage}")
    print(f"  结束原因: {result.finish_reason}")


# ═══════════════════════════════════════════════════════════
# 3. 多模型对比调用（同一问题，不同模型）
# ═══════════════════════════════════════════════════════════

def demo_multi_model_comparison():
    """多模型对比：同一问题发给所有已配置模型"""
    print_separator("3. 多模型对比调用")

    question = "用一句话解释什么是大模型的'幻觉'问题？"
    messages = [
        {"role": "system", "content": "你是AI领域专家，回答简洁。"},
        {"role": "user", "content": question},
    ]

    print(f"  统一问题: {question}\n")

    for model_name in cfg.available_models:
        model_cfg = cfg.get_model(model_name)
        print(f"  [{model_name}] (格式: {model_cfg.provider}, 模型: {model_cfg.model})")

        try:
            client = create_client(model_name)
            result = client.chat(messages)
            print(f"    回答: {result.content}")
            print(f"    Token: {result.usage}")
        except Exception as e:
            print(f"    ❌ 调用失败: {e}")

        print()


# ═══════════════════════════════════════════════════════════
# 4. 多轮对话（手动维护历史）
# ═══════════════════════════════════════════════════════════

def demo_multi_turn_manual():
    """
    多轮对话 — 手动维护 messages 历史

    核心原理：
        大模型 API 是无状态的（Stateless），服务器不保存任何聊天记录。
        所谓"多轮对话"，是客户端自己把历史消息累积起来，每次请求都带上完整历史。

    关键操作：
        每次 AI 回复后，必须把回复 append 到 messages 列表中，
        下一次请求时一起发送，AI 才能"记住"之前说过的话。
    """
    print_separator("4. 多轮对话（手动维护历史）")

    client = create_client()
    print(f"  客户端: {client}\n")

    # 初始化对话历史
    messages = [
        {"role": "system", "content": "你是一个友好的AI助手，回答简洁。"},
    ]

    # 模拟 3 轮对话
    user_questions = [
        "我叫小明，我是一名Python开发者。",
        "我叫什么名字？我的职业是什么？",
        "推荐一个适合我职业的学习资源。",
    ]

    for i, question in enumerate(user_questions, 1):
        print(f"  [第 {i} 轮]")
        print(f"  用户: {question}")

        # 1. 添加用户消息到历史
        messages.append({"role": "user", "content": question})

        # 2. 调用 API（带完整历史）
        result = client.chat(messages)

        # 3. 打印回复
        print(f"  助手: {result.content}")
        print(f"  (prompt_tokens: {result.usage['prompt_tokens'] if result.usage else '?'})")

        # 4. 关键：把 AI 回复加回历史（下一轮才能接上下文）
        messages.append({"role": "assistant", "content": result.content})

        print()

    print(f"  最终 messages 长度: {len(messages)} 条")
    print("  注意观察 prompt_tokens 随轮数增加而增长（成本线性增长）")


# ═══════════════════════════════════════════════════════════
# 5. 多轮对话（使用 ContextManager）
# ═══════════════════════════════════════════════════════════

def demo_multi_turn_with_context_manager():
    """
    多轮对话 — 使用 ContextManager 自动管理上下文

    优势：
        - 自动应用上下文管理策略（滑动窗口/Token预算等）
        - 自动处理 reasoning_content 的回传规则
        - 提供 token 估算和统计
    """
    print_separator("5. 多轮对话（ContextManager 管理）")

    client = create_client()

    # 创建上下文管理器（使用滑动窗口策略，最多保留 5 轮）
    cm = ContextManager(
        system_prompt="你是一个耐心的编程导师，擅长用简单比喻解释复杂概念。",
        strategy=SlidingWindowStrategy(max_rounds=5),
    )

    print(f"  策略: {cm.strategy.name}")
    print(f"  客户端: {client}\n")

    # 模拟多轮对话
    questions = [
        "什么是递归？",
        "能举个生活中的例子吗？",
        "递归和循环有什么区别？",
        "什么时候该用递归而不是循环？",
    ]

    for i, q in enumerate(questions, 1):
        print(f"  [第 {i} 轮] 用户: {q}")

        # 添加用户消息
        cm.add_user_message(q)

        # 获取处理后的 messages（自动应用策略）
        messages = cm.get_messages()

        # 调用 API
        result = client.chat(messages)

        # 添加助手回复到历史
        cm.add_assistant_message(
            content=result.content,
            reasoning=result.reasoning,
        )

        print(f"  助手: {result.content[:100]}{'...' if len(result.content or '') > 100 else ''}")
        print(f"  [历史: {cm.message_count} 条, 估算 tokens: {cm.get_token_estimate()}]")
        print()

    print(f"  最终状态: {cm}")


# ═══════════════════════════════════════════════════════════
# 6. 流式多轮对话
# ═══════════════════════════════════════════════════════════

def demo_stream_multi_turn():
    """流式输出 + 多轮对话"""
    print_separator("6. 流式多轮对话")

    client = create_client()
    print(f"  客户端: {client}\n")

    messages = [
        {"role": "system", "content": "你是一个简洁的助手。"},
        {"role": "user", "content": "用两句话介绍机器学习。"},
    ]

    print("  [第 1 轮] 用户: 用两句话介绍机器学习。")
    print("  助手: ", end="")

    # 流式调用
    result = client.chat_stream_print(messages)

    # 把回复加入历史
    messages.append({"role": "assistant", "content": result.content})

    # 第二轮
    messages.append({"role": "user", "content": "它和深度学习是什么关系？"})
    print("\n  [第 2 轮] 用户: 它和深度学习是什么关系？")
    print("  助手: ", end="")

    result2 = client.chat_stream_print(messages)
    print()


# ═══════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "╔" + "═"*58 + "╗")
    print("║   01_多环境多模型多轮对话 — 生产级 LLM 调用演示         ║")
    print("╚" + "═"*58 + "╝")

    # 1. 环境信息
    demo_env_info()

    # 2. 单模型调用
    demo_single_call()

    # 3. 多模型对比（取消注释运行）
    # demo_multi_model_comparison()

    # 4. 多轮对话（手动维护）
    demo_multi_turn_manual()

    # 5. 多轮对话（ContextManager）
    demo_multi_turn_with_context_manager()

    # 6. 流式多轮对话（取消注释运行）
    # demo_stream_multi_turn()

    print_separator("演示完成")
