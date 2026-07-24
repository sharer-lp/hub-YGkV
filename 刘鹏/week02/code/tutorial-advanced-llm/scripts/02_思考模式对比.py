"""
02_思考模式对比.py

==================== 功能说明 ====================

对比不同模型在"思考"与"不思考"模式下的参数差异、响应差异和适用场景。

核心知识点：
    1. 各厂商思考控制参数对比：
       - DeepSeek: extra_body={"thinking": {"type": "enabled/disabled"}}
       - OpenAI:   reasoning_effort="low/medium/high"
       - Anthropic: thinking={"type": "enabled", "budget_tokens": N}
       - Qwen/Kimi: 暂无原生思考控制参数

    2. 思考模式下的响应结构：
       - reasoning_content: 思考过程（草稿纸）
       - content: 最终回答（卷子上的答案）

    3. 思考模式的限制：
       - 不支持 temperature / top_p 参数
       - 思考内容也消耗 token（计费）
       - max_tokens 包含思考内容（需设大）

    4. 适用场景对比：
       - 思考模式：数学推理、逻辑分析、代码生成、复杂决策
       - 普通模式：简单翻译、闲聊、改写、快速问答

运行方式：
    cd tutorial-advanced-llm
    python scripts/02_思考模式对比.py

==================================================
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from advanced_llm import create_client, cfg


def print_separator(title: str = ""):
    if title:
        print(f"\n{'='*65}")
        print(f"  {title}")
        print(f"{'='*65}")
    else:
        print(f"\n{'-'*65}")


def print_thinking_params_comparison():
    """打印各厂商思考控制参数对比表"""
    print_separator("各厂商思考控制参数对比")

    comparison = """
    ┌─────────────┬──────────────────────────────────────────────────────────────┐
    │ 厂商/模型    │ 思考控制方式                                                  │
    ├─────────────┼──────────────────────────────────────────────────────────────┤
    │ DeepSeek    │ extra_body={"thinking": {"type": "enabled"/"disabled"}}       │
    │             │ + reasoning_effort="low/medium/high"（控制思考深度）            │
    │             │ 特点：无原生 budget 参数，思考深度由模型训练决定                  │
    ├─────────────┼──────────────────────────────────────────────────────────────┤
    │ OpenAI      │ reasoning_effort="low/medium/high"（粗粒度等级）               │
    │ (o3/o4)    │ 特点：不可指定具体 token 数，只能选等级                          │
    ├─────────────┼──────────────────────────────────────────────────────────────┤
    │ Anthropic   │ thinking={"type": "enabled", "budget_tokens": 4096}           │
    │ (Claude)    │ 特点：精确 token 预算控制，可限制思考长度                        │
    ├─────────────┼──────────────────────────────────────────────────────────────┤
    │ Qwen        │ 暂无原生思考控制参数                                           │
    │ (通义千问)   │ 可通过 prompt 引导"请逐步思考"实现类似效果                      │
    ├─────────────┼──────────────────────────────────────────────────────────────┤
    │ Kimi        │ 暂无原生思考控制参数                                           │
    │ (Moonshot)  │ 可通过 prompt 引导实现                                         │
    └─────────────┴──────────────────────────────────────────────────────────────┘

    【关键差异总结】
    1. 参数命名不统一：DeepSeek 用 thinking.type，Anthropic 用 thinking.budget_tokens
    2. 控制粒度不同：Anthropic 精确到 token 数，OpenAI 只有等级
    3. 副作用一致：思考模式下均不支持 temperature/top_p
    4. 计费一致：思考内容均计入 completion_tokens（花钱）
    """
    print(comparison)


# ═══════════════════════════════════════════════════════════
# 1. 同一模型：思考 vs 不思考 对比
# ═══════════════════════════════════════════════════════════

def demo_thinking_vs_no_thinking():
    """
    对比同一模型在思考/不思考模式下的差异

    观察点：
        - 响应时间差异（思考模式更慢）
        - Token 消耗差异（思考模式消耗更多）
        - 回答质量差异（复杂推理题思考模式更准确）
        - 响应结构差异（思考模式多一个 reasoning_content 字段）
    """
    print_separator("1. 思考 vs 不思考 — 同一问题对比")

    # 选择一个需要推理的问题
    question = "一个房间里有3个人，走了2个人，又来了5个人，现在房间里有几个人？请仔细思考。"

    print(f"  问题: {question}\n")

    # 仅对支持思考的模型进行对比
    model_name = cfg.active_model.name
    model_cfg = cfg.active_model

    if not model_cfg.supports_thinking:
        print(f"  ⚠️  当前模型 [{model_name}] 不支持原生思考控制")
        print(f"      支持的模型: deepseek, claude")
        print(f"      将演示普通模式调用\n")

    client = create_client()

    # --- 不思考模式 ---
    print("  【不思考模式】")
    start = time.time()
    result_no_think = client.chat(
        [{"role": "user", "content": question}],
        enable_thinking=False,
    )
    elapsed_no_think = time.time() - start

    print(f"    回答: {result_no_think.content}")
    print(f"    思考内容: {'无' if not result_no_think.reasoning else result_no_think.reasoning[:100] + '...'}")
    print(f"    耗时: {elapsed_no_think:.2f}s")
    print(f"    Token: {result_no_think.usage}")

    # --- 思考模式 ---
    if model_cfg.supports_thinking:
        print(f"\n  【思考模式】")
        start = time.time()
        result_think = client.chat(
            [{"role": "user", "content": question}],
            enable_thinking=True,
        )
        elapsed_think = time.time() - start

        print(f"    回答: {result_think.content}")
        if result_think.reasoning:
            print(f"    思考过程: {result_think.reasoning[:200]}...")
        else:
            print(f"    思考过程: (模型未输出思考内容)")
        print(f"    耗时: {elapsed_think:.2f}s")
        print(f"    Token: {result_think.usage}")

        # --- 对比总结 ---
        print(f"\n  【对比总结】")
        print(f"    耗时差异: {elapsed_think:.2f}s vs {elapsed_no_think:.2f}s "
              f"(思考模式慢 {elapsed_think/max(elapsed_no_think, 0.01):.1f}x)")

        if result_think.usage and result_no_think.usage:
            think_tokens = result_think.usage.get("completion_tokens", 0)
            no_think_tokens = result_no_think.usage.get("completion_tokens", 0)
            print(f"    输出Token: {think_tokens} vs {no_think_tokens} "
                  f"(思考模式多 {think_tokens - no_think_tokens} tokens)")
    else:
        print(f"\n  【跳过思考模式】当前模型不支持")


# ═══════════════════════════════════════════════════════════
# 2. 不同复杂度问题的思考效果对比
# ═══════════════════════════════════════════════════════════

def demo_thinking_by_complexity():
    """
    不同复杂度问题下思考模式的效果对比

    结论：
        - 简单问题：思考模式收益小，反而增加延迟和成本
        - 复杂推理：思考模式显著提升准确率
    """
    print_separator("2. 不同复杂度问题的思考效果")

    questions = [
        ("简单", "1+1等于几？"),
        ("中等", "如果所有的猫都是动物，所有的动物都会呼吸，那么所有的猫都会呼吸吗？"),
        ("复杂", "鸡兔同笼：笼中有头35个，脚94只，问鸡兔各几何？请列方程求解。"),
    ]

    client = create_client()
    model_cfg = cfg.active_model

    if not model_cfg.supports_thinking:
        print(f"  ⚠️  当前模型 [{model_cfg.name}] 不支持思考模式，仅演示普通模式\n")

    for level, question in questions:
        print(f"\n  [{level}] {question}")

        # 普通模式
        start = time.time()
        result_normal = client.chat(
            [{"role": "user", "content": question}],
            enable_thinking=False,
            max_tokens=500,
        )
        t_normal = time.time() - start
        print(f"    普通模式 ({t_normal:.2f}s): {result_normal.content[:80]}...")

        # 思考模式
        if model_cfg.supports_thinking:
            start = time.time()
            result_think = client.chat(
                [{"role": "user", "content": question}],
                enable_thinking=True,
                max_tokens=2000,
            )
            t_think = time.time() - start
            print(f"    思考模式 ({t_think:.2f}s): {result_think.content[:80]}...")
            if result_think.reasoning:
                print(f"    思考摘要: {result_think.reasoning[:60]}...")


# ═══════════════════════════════════════════════════════════
# 3. 思考模式参数限制演示
# ═══════════════════════════════════════════════════════════

def demo_thinking_param_constraints():
    """
    演示思考模式下的参数限制

    关键限制：
        1. 思考模式下不支持 temperature / top_p（DeepSeek 会报错）
        2. max_tokens 包含思考内容（设太小会截断回答）
        3. reasoning_effort 只在思考模式下生效
    """
    print_separator("3. 思考模式参数限制")

    client = create_client()
    model_cfg = cfg.active_model

    print("""
    【参数限制说明】

    1. temperature / top_p:
       - 不思考模式：可自由设置（0.0~2.0）
       - 思考模式：不支持设置（DeepSeek 会返回 400 错误）
       - 原因：思考过程需要确定性推理，随机性会破坏逻辑链

    2. max_tokens:
       - 包含思考内容 + 正式回答
       - 思考模式建议设 4096+（思考可能占 2000~3000 tokens）
       - 如果 finish_reason="length"，说明被截断了

    3. reasoning_effort（思考深度）:
       - "low": 快速思考，适合简单问题
       - "medium": 平衡模式
       - "high": 深度思考，适合复杂推理
       - 仅在 enable_thinking=True 时生效
    """)

    # 演示：不思考模式下可以使用 temperature
    print("  [演示] 不思考 + temperature=0.1:")
    result = client.chat(
        [{"role": "user", "content": "说一个冷笑话"}],
        enable_thinking=False,
        temperature=0.1,
    )
    print(f"    → {result.content[:60]}")

    # 演示：思考模式下 max_tokens 截断
    if model_cfg.supports_thinking:
        print(f"\n  [演示] 思考模式 + max_tokens=100（故意设小，观察截断）:")
        result = client.chat(
            [{"role": "user", "content": "详细解释量子计算的原理"}],
            enable_thinking=True,
            max_tokens=100,
        )
        print(f"    finish_reason: {result.finish_reason}")
        print(f"    回答: {result.content[:50] if result.content else '(被截断，无回答)'}")
        if result.finish_reason == "length":
            print(f"    ⚠️  被 max_tokens 截断！思考内容占满了额度。")


# ═══════════════════════════════════════════════════════════
# 4. 流式思考过程展示
# ═══════════════════════════════════════════════════════════

def demo_stream_thinking():
    """
    流式输出思考过程

    流式下思考内容的获取方式：
        - 非流式：getattr(message, "reasoning_content", None) 一次性获取
        - 流式：getattr(delta, "reasoning_content", None) 逐 chunk 拼接

    chunk 顺序：
        1. 先输出 reasoning_content chunks（思考阶段）
        2. 再输出 content chunks（回答阶段）
        3. 最后 finish_reason + usage
    """
    print_separator("4. 流式思考过程展示")

    client = create_client()
    model_cfg = cfg.active_model

    if not model_cfg.supports_thinking:
        print(f"  ⚠️  当前模型不支持思考模式，演示普通流式输出\n")

    messages = [{"role": "user", "content": "9.11 和 9.8 哪个大？为什么？"}]

    print(f"  问题: {messages[0]['content']}\n")
    print("  [流式输出开始]")

    result = client.chat_stream_print(
        messages,
        enable_thinking=model_cfg.supports_thinking,
    )

    print(f"\n  [流式输出结束]")
    print(f"  最终回答: {result.content}")
    if result.reasoning:
        print(f"  思考长度: {len(result.reasoning)} 字符")
    print(f"  Token: {result.usage}")


# ═══════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "╔" + "═"*63 + "╗")
    print("║   02_思考模式对比 — 思考 vs 不思考的参数与效果差异          ║")
    print("╚" + "═"*63 + "╝")

    # 0. 参数对比表
    print_thinking_params_comparison()

    # 1. 思考 vs 不思考
    demo_thinking_vs_no_thinking()

    # 2. 不同复杂度问题
    demo_thinking_by_complexity()

    # 3. 参数限制
    demo_thinking_param_constraints()

    # 4. 流式思考（取消注释运行）
    # demo_stream_thinking()

    print_separator("演示完成")
