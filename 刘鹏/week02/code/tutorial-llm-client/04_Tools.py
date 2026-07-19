"""
工具调用 (Tools / Function Calling) — 让模型调用外部函数

API 参考：https://platform.openai.com/docs/guides/function-calling
"""

import json
import math
from openai import OpenAI

client = OpenAI(
    api_key="sk-879e6628fec6417cbfd6b69c3e4d6ac0",
    base_url="https://api.deepseek.com",
)

# ═════════════════════════════════════════════════════════════════════════════
# 0. 定义本地工具函数
# ═════════════════════════════════════════════════════════════════════════════

def get_weather(city: str, date: str = "") -> str:
    """模拟查询天气（实际应调用真实 API）"""
    data = {
        ("北京", "2026-07-13"): "晴，28~35°C，南风 2 级",
        ("北京", "2026-07-14"): "多云转阴，26~33°C，可能有阵雨",
        ("上海", "2026-07-13"): "小雨，25~30°C，东南风 3 级",
        ("上海", "2026-07-14"): "阴，24~29°C",
        ("深圳", "2026-07-13"): "雷阵雨，26~32°C",
    }
    return data.get((city, date), f"{city} {date} 的天气信息暂未收录。")


# ---------- 数学计算 ----------

def calculate(expression: str) -> str:
    """安全执行数学表达式（支持 ^ 作为幂运算）"""
    allowed = {"abs", "round", "max", "min", "sum", "pow", "sqrt", "pi", "e",
               "sin", "cos", "tan", "log", "log10"}
    # 将用户习惯的 ^ 替换为 Python 的 **
    expr = expression.replace("^", "**")
    try:
        result = eval(expr, {"__builtins__": {}}, {k: getattr(math, k, None) for k in allowed})
        return str(result)
    except Exception as e:
        return f"计算错误：{e}"


# ---------- 文档查询 ----------

def search_docs(keyword: str) -> str:
    """模拟内部文档检索"""
    knowledge_base = {
        "退款政策": "用户可在购买后 7 天内申请无理由退款。",
        "发货时间": "现货商品 48 小时内发货，预售商品以页面标注为准。",
        "会员等级": "普通会员、银卡会员、金卡会员、钻石会员，消费越多等级越高。",
    }
    return knowledge_base.get(keyword, f"未找到「{keyword}」相关文档。")


# ═════════════════════════════════════════════════════════════════════════════
# 工具描述 schema（传给模型）
# ═════════════════════════════════════════════════════════════════════════════

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市在指定日期的天气情况",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如 北京、上海",
                    },
                    "date": {
                        "type": "string",
                        "description": "日期，格式 YYYY-MM-DD，默认为今天",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "执行数学计算，支持四则运算和 math 库函数",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，如 2+2、sqrt(144)、sin(pi/2)",
                    },
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "在内部知识库中搜索相关文档",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "搜索关键词，如 退款政策、发货时间",
                    },
                },
                "required": ["keyword"],
            },
        },
    },
]

# 工具名 → 本地函数映射
FUNCTION_MAP = {
    "get_weather": get_weather,
    "calculate": calculate,
    "search_docs": search_docs,
}


def run_tool_call(tc) -> str:
    """执行一次工具调用，返回结果字符串。"""
    name = tc.function.name
    args = json.loads(tc.function.arguments)
    print(f"    → 调用工具: {name}({json.dumps(args, ensure_ascii=False)})")
    result = FUNCTION_MAP[name](**args)
    print(f"    ← 结果: {result}")
    return result


# ═════════════════════════════════════════════════════════════════════════════
# 1. 单工具调用 — 模型自动选择工具
# ═════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("1️⃣  单工具调用 — 天气查询")
print("=" * 65)

messages = [
    {"role": "system", "content": "你是智能客服助手，可根据需要使用工具回答用户问题。当前日期：2026-07-13。"},
    {"role": "user", "content": "北京今天天气怎么样？"},
]

response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=messages,
    tools=TOOLS,
    temperature=0.0,
)

choice = response.choices[0]
msg = choice.message

# 模型可能直接回复（不需要工具），也可能发起工具调用
if msg.tool_calls:
    for tc in msg.tool_calls:
        result = run_tool_call(tc)
        messages.append(msg)                     # 保留 assistant 的 tool_calls
        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": result,
        })

    # 把工具结果发回模型，让其生成最终回复
    final = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=messages,
        tools=TOOLS,
        temperature=0.0,
    )
    print(f"\n最终回复: {final.choices[0].message.content}")
else:
    print(f"直接回复: {msg.content}")

print()


# ═════════════════════════════════════════════════════════════════════════════
# 2. 多工具连续调用 — 模型先调一个工具，根据结果再调另一个
# ═════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("2️⃣  多工具连续调用 — 查询 + 计算")
print("=" * 65)

messages = [
    {"role": "system", "content": "你是智能客服助手，可根据需要使用工具回答用户问题。当前日期：2026-07-13。"},
    {"role": "user", "content": "北京今天气温范围是多少？最高温和最低温的平均值是多少？"},
]

# 循环处理：模型可能一次返回多个 tool_call，也可能逐次返回
max_turns = 5
for turn in range(max_turns):
    print(f"\n  --- 第 {turn + 1} 轮 ---")

    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=messages,
        tools=TOOLS,
        temperature=0.0,
    )

    choice = response.choices[0]
    print(choice)
    msg = choice.message

    if msg.tool_calls:
        for tc in msg.tool_calls:
            result = run_tool_call(tc)
            messages.append(msg)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })
    else:
        # 没有工具调用了，输出最终回复
        print(f"  → 最终回复: {msg.content}")
        break
else:
    print(f"\n⚠️  达到最大轮数 {max_turns}，未完全收敛。")

print()


# ═════════════════════════════════════════════════════════════════════════════
# 3. 并行工具调用 — 模型一次请求多个工具（取决于模型能力）
# ═════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("3️⃣  并行工具调用 — 同时查多个信息")
print("=" * 65)

messages = [
    {"role": "system", "content": "你是智能客服助手，可根据需要使用工具回答用户问题。当前日期：2026-07-13。"},
    {"role": "user", "content": "我想查一下退款政策，还有 25 * 48 等于多少？"},
]

response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=messages,
    tools=TOOLS,
    temperature=0.0,
)

choice = response.choices[0]
print(choice)
msg = choice.message

if msg.tool_calls:
    print(f"  模型发起了 {len(msg.tool_calls)} 个工具调用：\n")

    # 并行执行所有工具
    results = []
    for tc in msg.tool_calls:
        result = run_tool_call(tc)
        results.append({"tool_call_id": tc.id, "result": result})

    # 一次把所有结果加回去
    messages.append(msg)
    for r in results:
        messages.append({
            "role": "tool",
            "tool_call_id": r["tool_call_id"],
            "content": r["result"],
        })

    final = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=messages,
        tools=TOOLS,
        temperature=0.0,
    )
    print(f"\n最终回复: {final.choices[0].message.content}")
else:
    print(f"直接回复: {msg.content}")

print()


# ═════════════════════════════════════════════════════════════════════════════
# 4. 强制使用特定工具 (tool_choice)
# ═════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("4️⃣  强制指定工具 — tool_choice")
print("=" * 65)

# tool_choice="auto"          — 模型自行选择（默认）
# tool_choice="none"          — 禁止使用工具
# tool_choice={"type":"function","function":{"name":"calculate"}} — 强制调用

print("  a) tool_choice='none' — 禁止工具（即使问题需要）")
response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[
        {"role": "system", "content": "你是智能客服助手。"},
        {"role": "user", "content": "请查询上海的天气。如果你没有天气数据，请告知。"},
    ],
    tools=TOOLS,
    tool_choice="none",
    temperature=0.0,
)
print(f"     → {response.choices[0].message.content}\n")

print("  b) 强制调用 calculate")
print("     ⚠️  部分模型（如 DeepSeek）不支持 thinking 模式下强制指定工具")
print("     下面尝试不带 thinking 模式的调用：")

response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[
        {"role": "system", "content": "你是智能客服助手。"},
        {"role": "user", "content": "请问 123 * 456 等于多少？"},
    ],
    tools=TOOLS,
    tool_choice={"type": "function", "function": {"name": "calculate"}},
    temperature=0.0,
    extra_body={"thinking": {"type": "disabled"}},
)
msg = response.choices[0].message
if msg.tool_calls:
    for tc in msg.tool_calls:
        result = run_tool_call(tc)
    print("     （强制工具模式下模型只返回 tool_call，无文本回复）")
else:
    print(f"     → {msg.content}")

print()


# ═════════════════════════════════════════════════════════════════════════════
# 5. 流式工具调用（stream=True 时处理 tool_calls）
# ═════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("5️⃣  流式工具调用")
print("=" * 65)

messages = [
    {"role": "system", "content": "你是智能客服助手，可根据需要使用工具回答用户问题。"},
    {"role": "user", "content": "帮我算一下 2^10 等于多少？"},
]

stream = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=messages,
    tools=TOOLS,
    stream=True,
    temperature=0.0,
)

# 流式下需要手动拼接 tool_calls
tool_calls = {}
final_content = ""

for chunk in stream:
    choice = chunk.choices[0] if chunk.choices else None
    if not choice:
        continue

    delta = choice.delta

    # 累积 tool_calls
    if delta.tool_calls:
        for tc_delta in delta.tool_calls:
            idx = tc_delta.index
            if idx not in tool_calls:
                tool_calls[idx] = tc_delta
            else:
                # 追加内容
                existing = tool_calls[idx]
                if tc_delta.function and tc_delta.function.name:
                    existing.function.name = (existing.function.name or "") + tc_delta.function.name
                if tc_delta.function and tc_delta.function.arguments:
                    existing.function.arguments = (existing.function.arguments or "") + tc_delta.function.arguments
                if tc_delta.id:
                    existing.id = tc_delta.id

    # 累积普通文本
    if delta.content:
        final_content += delta.content
        print(delta.content, end="", flush=True)

# 流式结束后，处理 tool_calls
if tool_calls:
    print("\n  （流中检测到工具调用，开始执行...）\n")
    messages.append({
        "role": "assistant",
        "content": None,
        "tool_calls": [tc.model_dump() for tc in tool_calls.values()],
    })

    for idx in sorted(tool_calls):
        tc = tool_calls[idx]
        tool_name = tc.function.name
        tool_args = json.loads(tc.function.arguments)
        print(f"  → 调用: {tool_name}({json.dumps(tool_args, ensure_ascii=False)})")
        result = FUNCTION_MAP[tool_name](**tool_args)
        print(f"  ← 结果: {result}")
        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": result,
        })

    final = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=messages,
        tools=TOOLS,
        temperature=0.0,
    )
    print(f"\n最终回复: {final.choices[0].message.content}")
elif final_content:
    print("\n（模型直接回复，无工具调用）\n")
