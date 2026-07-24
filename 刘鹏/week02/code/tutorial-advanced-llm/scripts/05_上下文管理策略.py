"""
05_上下文管理策略.py

==================== 功能说明 ====================

演示生产环境中常用的上下文（Context）管理策略，解决多轮对话中：
    1. 上下文窗口溢出问题
    2. Token 成本线性增长问题
    3. 长对话信息丢失问题

四种策略对比：
    ┌─────────────────────┬──────────┬──────────┬──────────────────────────┐
    │ 策略                │ 成本     │ 信息保留 │ 适用场景                 │
    ├─────────────────────┼──────────┼──────────┼──────────────────────────┤
    │ 滑动窗口截断        │ 低       │ 低       │ 闲聊、简单问答           │
    │ Token 预算截断      │ 中       │ 中       │ 精确控制成本的生产环境   │
    │ 摘要压缩            │ 高       │ 高       │ 长对话、客服、教育       │
    │ 混合策略（推荐）    │ 中       │ 高       │ 大多数生产环境           │
    └─────────────────────┴──────────┴──────────┴──────────────────────────┘

运行方式：
    cd tutorial-advanced-llm
    python scripts/05_上下文管理策略.py

==================================================
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from advanced_llm import create_client, cfg
from advanced_llm.context_manager import (
    ContextManager,
    SlidingWindowStrategy,
    TokenBudgetStrategy,
    SummaryStrategy,
    HybridStrategy,
    estimate_tokens,
    estimate_messages_tokens,
)


def print_separator(title: str = ""):
    if title:
        print(f"\n{'='*65}")
        print(f"  {title}")
        print(f"{'='*65}")
    else:
        print(f"\n{'-'*65}")


# ═══════════════════════════════════════════════════════════
# 0. 问题背景：为什么需要上下文管理？
# ═══════════════════════════════════════════════════════════

def demo_why_context_management():
    """
    演示不做上下文管理时的问题

    核心问题：
        1. 成本爆炸：10轮对话的总 token 消耗是单轮的 6 倍
        2. 窗口溢出：超过模型上下文窗口直接报错
        3. 速度下降：prompt 越长，首 token 延迟越高
    """
    print_separator("0. 为什么需要上下文管理？")

    print("""
    【问题本质】
    大模型 API 是无状态的，每次请求必须携带完整对话历史。
    随着轮数增加，历史消息不断累积：

    轮次    发送 token    累计消耗
    ─────────────────────────────
    第1轮    100           100
    第2轮    200           300
    第3轮    300           600
    第5轮    500           1500
    第10轮   1000          5500
    第20轮   2000          21000  ← 成本爆炸！

    【三大风险】
    1. 成本风险：长对话单次请求可能消耗数千 token（几毛到几块钱）
    2. 窗口溢出：模型上下文窗口有限（如 128K），超过直接 400 错误
    3. 延迟增加：prompt 越长，模型处理越慢（首 token 延迟线性增长）

    【解决方案】
    → 上下文管理策略：在保留关键信息的前提下，控制历史长度
    """)


# ═══════════════════════════════════════════════════════════
# 1. 策略一：滑动窗口截断
# ═══════════════════════════════════════════════════════════

def demo_sliding_window():
    """
    滑动窗口截断策略

    原理：只保留最近 N 轮对话，丢弃更早的消息
    优点：实现简单，零额外 API 开销
    缺点：会丢失早期重要信息（如用户姓名、偏好）

    适用：闲聊、简单问答、不依赖长期记忆的场景
    """
    print_separator("1. 滑动窗口截断策略")

    # 创建策略（最多保留 3 轮）
    strategy = SlidingWindowStrategy(max_rounds=3)
    cm = ContextManager(
        system_prompt="你是友好的助手。",
        strategy=strategy,
    )

    print(f"  策略: {strategy.name}")
    print(f"  规则: 只保留最近 3 轮对话\n")

    # 模拟 6 轮对话
    conversations = [
        ("我叫小明，我是Python开发者。", "你好小明！Python开发者很棒。"),
        ("我最喜欢用FastAPI框架。", "FastAPI确实很好用，性能优秀。"),
        ("我住在北京。", "北京是个好城市！"),
        ("我叫什么名字？", "你叫小明。"),
        ("我住在哪里？", "你住在北京。"),
        ("我最喜欢什么框架？", "你最喜欢FastAPI框架。"),
    ]

    for i, (user_msg, expected_reply) in enumerate(conversations, 1):
        cm.add_user_message(user_msg)
        cm.add_assistant_message(expected_reply)

        messages = cm.get_messages()
        tokens = estimate_messages_tokens(messages)

        print(f"  第{i}轮: \"{user_msg}\"")
        print(f"    历史消息数: {len(messages)}, 估算tokens: {tokens}")

        if i == 4:
            print(f"    ⚠️  注意：第1轮（姓名信息）可能已被截断！")
        print()

    print(f"  最终状态: {cm}")
    print(f"\n  【分析】")
    print(f"  - 滑动窗口保留了最近 3 轮，早期信息被丢弃")
    print(f"  - 第4轮问'我叫什么'时，如果第1轮被截断，模型将无法回答")
    print(f"  - 适合不依赖早期信息的场景")


# ═══════════════════════════════════════════════════════════
# 2. 策略二：Token 预算截断
# ═══════════════════════════════════════════════════════════

def demo_token_budget():
    """
    Token 预算截断策略

    原理：按 token 数（而非消息数）精确控制上下文大小
    优点：精确控制成本，不会超出模型窗口
    缺点：可能在不完整位置截断

    适用：需要精确控制 token 消耗的生产环境
    """
    print_separator("2. Token 预算截断策略")

    # 创建策略（最大 500 token，预留 200 给输出）
    strategy = TokenBudgetStrategy(max_tokens=500, reserve_tokens=200)
    cm = ContextManager(
        system_prompt="你是助手。",
        strategy=strategy,
    )

    print(f"  策略: {strategy.name}")
    print(f"  规则: 上下文最多 500 tokens，预留 200 给输出\n")

    # 模拟对话（每轮内容较长）
    long_conversations = [
        "请详细解释Python中的装饰器是什么，它的工作原理是什么，以及在实际开发中有哪些常见的应用场景？",
        "装饰器本质上是一个接受函数作为参数并返回新函数的高阶函数。它利用Python的闭包特性，在不修改原函数代码的前提下扩展函数功能。常见应用包括：日志记录、性能计时、权限验证、缓存等。",
        "那上下文管理器呢？with语句是怎么工作的？",
        "上下文管理器通过__enter__和__exit__方法管理资源的获取和释放。with语句确保即使发生异常，资源也能被正确释放。",
        "能比较一下装饰器和上下文管理器的异同吗？",
    ]

    for i, msg in enumerate(long_conversations):
        role = "user" if i % 2 == 0 else "assistant"
        if role == "user":
            cm.add_user_message(msg)
        else:
            cm.add_assistant_message(msg)

        raw_tokens = estimate_messages_tokens(cm.get_raw_messages())
        managed_messages = cm.get_messages()
        managed_tokens = estimate_messages_tokens(managed_messages)

        print(f"  第{i+1}条 ({role}): \"{msg[:30]}...\"")
        print(f"    原始tokens: {raw_tokens} → 管理后: {managed_tokens} "
              f"(消息数: {len(cm.get_raw_messages())} → {len(managed_messages)})")

        if raw_tokens > managed_tokens:
            print(f"    ✂️  截断了 {raw_tokens - managed_tokens} tokens")
        print()

    print(f"  【分析】")
    print(f"  - Token 预算策略精确控制总 token 数不超过 500")
    print(f"  - 从最早的消息开始删除，保留最近的对话")
    print(f"  - 比滑动窗口更精确（按实际 token 数而非消息数）")


# ═══════════════════════════════════════════════════════════
# 3. 策略三：摘要压缩
# ═══════════════════════════════════════════════════════════

def demo_summary_strategy():
    """
    摘要压缩策略

    原理：当历史超过阈值时，用模型将早期对话总结为摘要
    优点：信息保留好，能记住早期关键信息
    缺点：需要额外 API 调用（增加延迟和成本）

    适用：长对话、客服、教育辅导等需要记住用户信息的场景
    """
    print_separator("3. 摘要压缩策略")

    client = create_client()

    # 创建策略
    strategy = SummaryStrategy(
        max_rounds=3,          # 保留最近 3 轮原文
        summary_trigger=4,     # 超过 4 轮时触发摘要
        client=client,         # 用于生成摘要的客户端
    )
    cm = ContextManager(
        system_prompt="你是耐心的助手。",
        strategy=strategy,
    )

    print(f"  策略: {strategy.name}")
    print(f"  规则: 超过 4 轮时，将早期对话压缩为摘要\n")

    # 模拟多轮对话
    conversations = [
        ("我叫张三，是一名数据科学家。", "你好张三！数据科学家是很棒的职业。"),
        ("我主要用Python和SQL做数据分析。", "Python和SQL是数据分析的核心工具。"),
        ("我在一家金融公司工作。", "金融数据分析很有挑战性！"),
        ("我最近在学习深度学习。", "深度学习在金融领域有很多应用。"),
        ("我叫什么？在哪里工作？", None),  # 测试是否记住早期信息
    ]

    for i, (user_msg, assistant_reply) in enumerate(conversations, 1):
        cm.add_user_message(user_msg)

        if assistant_reply:
            cm.add_assistant_message(assistant_reply)
            print(f"  第{i}轮: \"{user_msg}\" → \"{assistant_reply}\"")
        else:
            # 实际调用 API 获取回答
            messages = cm.get_messages()
            print(f"\n  第{i}轮: \"{user_msg}\"")
            print(f"  (触发摘要后的 messages 结构:)")
            for m in messages:
                role = m['role']
                content = (m.get('content') or '')[:50]
                print(f"    [{role}] {content}...")

            result = client.chat(messages)
            cm.add_assistant_message(result.content)
            print(f"  模型回答: {result.content}")

        print(f"    消息数: {cm.message_count}, tokens: {cm.get_token_estimate()}")
        print()

    print(f"  【分析】")
    print(f"  - 早期对话被压缩为摘要（保留关键信息：姓名、职业、公司）")
    print(f"  - 最近 3 轮保留原文（保持对话连贯性）")
    print(f"  - 即使早期消息被压缩，模型仍能回答'我叫什么'")


# ═══════════════════════════════════════════════════════════
# 4. 策略四：混合策略（生产推荐）
# ═══════════════════════════════════════════════════════════

def demo_hybrid_strategy():
    """
    混合策略（生产环境推荐）

    原理：组合多种策略
        1. 先按轮数粗截断（滑动窗口）
        2. 再按 token 预算精确截断
        3. 可选：对截断部分生成摘要

    优点：兼顾成本控制和信息保留
    适用：大多数生产环境的首选
    """
    print_separator("4. 混合策略（生产推荐）")

    strategy = HybridStrategy(
        max_rounds=5,          # 滑动窗口：最多 5 轮
        max_tokens=2000,       # Token 预算：最多 2000
        reserve_tokens=500,    # 预留输出空间
    )
    cm = ContextManager(
        system_prompt="你是智能客服。",
        strategy=strategy,
    )

    print(f"  策略: {strategy.name}")
    print(f"  规则: 滑动窗口(5轮) + Token预算(2000)\n")

    # 模拟 8 轮对话
    for i in range(1, 9):
        user_msg = f"第{i}轮用户消息：这是模拟的对话内容，用于测试上下文管理策略的效果。"
        assistant_msg = f"第{i}轮助手回复：收到你的消息，这是模拟的回答。"

        cm.add_user_message(user_msg)
        cm.add_assistant_message(assistant_msg)

        raw_count = len(cm.get_raw_messages())
        managed = cm.get_messages()
        managed_count = len(managed)
        tokens = estimate_messages_tokens(managed)

        truncated = "✂️" if managed_count < raw_count else "  "
        print(f"  第{i}轮: 原始{raw_count}条 → 管理后{managed_count}条, "
              f"tokens≈{tokens} {truncated}")

    print(f"\n  最终状态: {cm}")
    print(f"\n  【混合策略优势】")
    print(f"  - 双重保障：轮数限制 + token 限制")
    print(f"  - 不会超出模型窗口")
    print(f"  - 成本可控且可预测")


# ═══════════════════════════════════════════════════════════
# 5. 实际 API 调用对比
# ═══════════════════════════════════════════════════════════

def demo_real_api_comparison():
    """
    用真实 API 调用对比不同策略下的 token 消耗

    观察：
        - 无管理：token 随轮数线性增长
        - 滑动窗口：token 稳定在固定范围
        - 混合策略：token 精确可控
    """
    print_separator("5. 真实 API 调用 — Token 消耗对比")

    client = create_client()

    # 模拟 5 轮对话内容
    test_conversations = [
        ("你好，我叫李四。", None),
        ("帮我解释一下什么是递归。", None),
        ("能举个Python代码例子吗？", None),
        ("递归有什么缺点？", None),
        ("我叫什么名字？", None),
    ]

    # --- 无管理 ---
    print("\n  【无上下文管理】")
    messages_raw = [{"role": "system", "content": "你是助手。"}]
    total_tokens_raw = 0

    for i, (q, _) in enumerate(test_conversations, 1):
        messages_raw.append({"role": "user", "content": q})
        result = client.chat(messages_raw)
        messages_raw.append({"role": "assistant", "content": result.content})

        prompt_tokens = result.usage.get("prompt_tokens", 0) if result.usage else 0
        total_tokens_raw += prompt_tokens
        print(f"    第{i}轮: prompt_tokens={prompt_tokens}")

    print(f"    总 prompt_tokens: {total_tokens_raw}")

    # --- 滑动窗口 ---
    print(f"\n  【滑动窗口(max_rounds=2)】")
    cm = ContextManager(
        system_prompt="你是助手。",
        strategy=SlidingWindowStrategy(max_rounds=2),
    )
    total_tokens_sw = 0

    for i, (q, _) in enumerate(test_conversations, 1):
        cm.add_user_message(q)
        messages = cm.get_messages()
        result = client.chat(messages)
        cm.add_assistant_message(result.content)

        prompt_tokens = result.usage.get("prompt_tokens", 0) if result.usage else 0
        total_tokens_sw += prompt_tokens
        print(f"    第{i}轮: prompt_tokens={prompt_tokens}, 消息数={len(messages)}")

    print(f"    总 prompt_tokens: {total_tokens_sw}")

    # 对比
    print(f"\n  【对比总结】")
    print(f"    无管理: {total_tokens_raw} tokens")
    print(f"    滑动窗口: {total_tokens_sw} tokens")
    if total_tokens_raw > 0:
        saving = (1 - total_tokens_sw / total_tokens_raw) * 100
        print(f"    节省: {saving:.0f}%")


# ═══════════════════════════════════════════════════════════
# 6. 策略选择指南
# ═══════════════════════════════════════════════════════════

def print_strategy_guide():
    """打印策略选择指南"""
    print_separator("6. 策略选择指南")

    print("""
    【根据场景选择策略】

    场景1: 简单问答 / 闲聊（不依赖历史）
    → SlidingWindowStrategy(max_rounds=5)
    → 理由：简单高效，零额外开销

    场景2: 精确成本控制（按 token 计费的业务）
    → TokenBudgetStrategy(max_tokens=4000, reserve_tokens=1000)
    → 理由：精确控制每次请求的 token 消耗

    场景3: 长对话客服 / 教育辅导（需记住用户信息）
    → SummaryStrategy(max_rounds=5, summary_trigger=8, client=client)
    → 理由：早期信息被摘要保留，不会"失忆"

    场景4: 大多数生产环境（推荐默认）
    → HybridStrategy(max_rounds=10, max_tokens=8000, reserve_tokens=2000)
    → 理由：双重保障，兼顾成本和信息保留

    场景5: Agent / 工具调用场景
    → HybridStrategy + 手动保留 reasoning_content
    → 理由：工具调用历史不能截断（会导致 API 报错）

    【生产环境最佳实践】
    1. 默认使用 HybridStrategy
    2. 工具调用轮次标记为"不可截断"
    3. 摘要异步生成（不阻塞主对话）
    4. 监控 token 消耗 P95，动态调整预算
    5. 会话存储用 Redis（TTL 24h），不要放内存
    """)


# ═══════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "╔" + "═"*63 + "╗")
    print("║   05_上下文管理策略 — 生产级多轮对话历史管理              ║")
    print("╚" + "═"*63 + "╝")

    # 0. 为什么需要上下文管理
    demo_why_context_management()

    # 1. 滑动窗口
    demo_sliding_window()

    # 2. Token 预算
    demo_token_budget()

    # 3. 摘要压缩（需要 API 调用）
    # demo_summary_strategy()

    # 4. 混合策略
    demo_hybrid_strategy()

    # 5. 真实 API 对比（需要 API 调用）
    # demo_real_api_comparison()

    # 6. 策略选择指南
    print_strategy_guide()

    print_separator("演示完成")
