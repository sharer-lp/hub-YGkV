"""
MCP Client — 连接 MCP Server 并通过 DeepSeek 调用工具

MCP 工作流程：
  1. Client 启动 MCP Server 子进程（通过 stdio 通信）
  2. 发现 Server 暴露的 tools / resources / prompts
  3. 将 MCP tools 转换为 OpenAI function calling 格式
  4. 用户提问 → DeepSeek 决策调用哪些工具 → 通过 MCP 执行 → 返回结果给模型 → 最终回复

与 04_Tools.py 的区别：
  - 04_Tools.py: 工具定义和执行都在本地代码中
  - 本文件:     工具定义和执行在远程 MCP Server 中，Client 只负责"发现"和"转发"

运行方式：
  pip install "mcp[cli]" openai
  python 06_MCP_Client.py

前置条件：
  确保 06_MCP_Server.py 与本文件在同一目录

调试技巧：
  - 如果连接失败，先单独运行 python 06_MCP_Server.py 确认 Server 无语法错误
  - 使用 mcp dev 06_MCP_Server.py 可以在浏览器中可视化调试 Server
"""

import asyncio
import json
import os
import sys

from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ═════════════════════════════════════════════════════════════════════════════
# 配置
# ═════════════════════════════════════════════════════════════════════════════

LLM_CLIENT = OpenAI(
    api_key="sk-879e6628fec6417cbfd6b69c3e4d6ac0",
    base_url="https://api.deepseek.com",
)

LLM_MODEL = "deepseek-v4-flash"

# MCP Server 的启动参数（指向同目录下的 Server 文件）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SERVER_SCRIPT = os.path.join(SCRIPT_DIR, "06_MCP_Server.py")


# ═════════════════════════════════════════════════════════════════════════════
# 辅助函数
# ═════════════════════════════════════════════════════════════════════════════

def mcp_tools_to_openai_tools(mcp_tools: list) -> list[dict]:
    """
    将 MCP 工具描述转换为 OpenAI function calling 格式。

    MCP 工具的 inputSchema 已经是 JSON Schema 格式，
    只需包装成 OpenAI 期望的结构即可。

    这是 MCP + LLM 集成的关键步骤：
    - MCP 定义了工具的标准描述格式
    - OpenAI 需要 function calling 格式
    - 这里做一层适配转换
    """
    openai_tools = []
    for tool in mcp_tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema,
            },
        })
    return openai_tools


async def call_tool(session: ClientSession, name: str, arguments: dict) -> str:
    """
    通过 MCP 协议调用远程工具。

    注意：这里不是本地函数调用，而是通过 JSON-RPC 发送请求给 Server 进程，
    Server 执行工具后返回结果。
    """
    result = await session.call_tool(name, arguments)
    # MCP 返回的 content 是 ContentBlock 列表，提取文本
    texts = []
    for block in result.content:
        if hasattr(block, "text"):
            texts.append(block.text)
    return "\n".join(texts)


# ═════════════════════════════════════════════════════════════════════════════
# 核心逻辑：Agent 循环（与 04_Tools.py 中的多轮调用逻辑一致）
# ═════════════════════════════════════════════════════════════════════════════

async def agent_loop(session: ClientSession, user_question: str, max_turns: int = 5):
    """
    Agent 循环：用户提问 → 模型决策 → MCP 执行工具 → 结果返回模型 → 最终回复

    这与 04_Tools.py 中的多轮工具调用逻辑一致，
    区别在于工具执行从本地函数变成了通过 MCP 协议的远程调用。
    """
    # 1. 发现 MCP Server 暴露的工具（自动发现，无需手动写 schema）
    mcp_tools = await session.list_tools()
    openai_tools = mcp_tools_to_openai_tools(mcp_tools.tools)

    print(f"  📋 发现 {len(openai_tools)} 个 MCP 工具:")
    for t in mcp_tools.tools:
        print(f"     - {t.name}: {(t.description or '')[:40]}...")

    # 2. 读取 MCP Resource（可选，用于增强上下文）
    resources = await session.list_resources()
    print(f"\n  📦 发现 {len(resources.resources)} 个 MCP 资源:")
    for r in resources.resources:
        print(f"     - {r.uri}: {r.name}")

    # 3. 构建消息列表
    messages = [
        {"role": "system", "content": "你是一个智能学习助手，可以使用各种工具帮助用户。请根据需要使用工具获取信息。"},
        {"role": "user", "content": user_question},
    ]

    print(f"\n  ❓ 用户问题: {user_question}\n")

    # 4. Agent 循环（最多 max_turns 轮）
    for turn in range(max_turns):
        print(f"  --- 第 {turn + 1} 轮 ---")

        # 调用 DeepSeek 模型
        response = LLM_CLIENT.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            tools=openai_tools if openai_tools else None,
            temperature=0.0,
        )

        msg = response.choices[0].message

        if msg.tool_calls:
            # 模型要求调用工具 → 通过 MCP 协议远程执行
            messages.append(msg)

            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                print(f"    → MCP 调用: {tc.function.name}({json.dumps(args, ensure_ascii=False)})")

                # 关键：通过 MCP 协议执行远程工具调用（而非本地函数）
                result = await call_tool(session, tc.function.name, args)
                print(f"    ← MCP 结果: {result[:100]}...")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
        else:
            # 模型生成了最终回复（不再需要工具）
            print(f"\n  ✅ 最终回复:\n  {msg.content}")
            return msg.content

    print(f"\n  ⚠️  达到最大轮数 {max_turns}")
    return None


# ═════════════════════════════════════════════════════════════════════════════
# 演示场景
# ═════════════════════════════════════════════════════════════════════════════

async def demo_basic_tool_call(session: ClientSession):
    """场景 1：基础工具调用 — 查询时间"""
    print("\n" + "=" * 65)
    print("1️⃣  基础 MCP 工具调用 — 查询当前时间")
    print("=" * 65)
    await agent_loop(session, "现在几点了？今天是星期几？")


async def demo_calculate(session: ClientSession):
    """场景 2：计算工具 — 数学计算"""
    print("\n" + "=" * 65)
    print("2️⃣  MCP 计算工具 — 数学计算")
    print("=" * 65)
    await agent_loop(session, "帮我算一下 2 的 16 次方等于多少？再帮我算 sin(π/4) 的值")


async def demo_stateful_tool(session: ClientSession):
    """场景 3：有状态工具 — 笔记本操作"""
    print("\n" + "=" * 65)
    print("3️⃣  MCP 有状态工具 — 笔记本管理")
    print("=" * 65)
    await agent_loop(session,
        "帮我记三条笔记：1) MCP是模型上下文协议 2) 它使用JSON-RPC通信 3) 支持tools、resources、prompts三种能力。"
        "然后列出所有笔记。")


async def demo_multi_tool(session: ClientSession):
    """场景 4：多工具协作 — 综合使用多个工具"""
    print("\n" + "=" * 65)
    print("4️⃣  多工具协作 — 综合使用多个工具")
    print("=" * 65)
    await agent_loop(session,
        "现在几点了？帮我算一下 2^8 等于多少，然后把计算结果记到笔记里。")


async def demo_resource_read(session: ClientSession):
    """场景 5：读取 MCP Resource（非工具调用，直接读取数据）"""
    print("\n" + "=" * 65)
    print("5️⃣  读取 MCP Resource — 获取服务器信息")
    print("=" * 65)

    # 列出所有可用资源
    resources = await session.list_resources()
    for r in resources.resources:
        print(f"\n  📦 读取资源: {r.uri} ({r.name})")
        # 直接读取资源内容（不经过 LLM，直接获取数据）
        result = await session.read_resource(r.uri)
        for block in result.contents:
            if hasattr(block, "text"):
                try:
                    data = json.loads(block.text)
                    print(f"     {json.dumps(data, ensure_ascii=False, indent=2)}")
                except json.JSONDecodeError:
                    print(f"     {block.text}")


async def demo_prompt_template(session: ClientSession):
    """场景 6：使用 MCP Prompt 模板"""
    print("\n" + "=" * 65)
    print("6️⃣  使用 MCP Prompt 模板 — 获取预定义提示词")
    print("=" * 65)

    # 列出所有可用 Prompt
    prompts = await session.list_prompts()
    print(f"\n  📝 发现 {len(prompts.prompts)} 个 Prompt 模板:")
    for p in prompts.prompts:
        print(f"     - {p.name}: {p.description}")

    # 获取具体的 Prompt 内容
    if prompts.prompts:
        prompt_name = prompts.prompts[0].name
        print(f"\n  获取 Prompt: {prompt_name}")
        result = await session.get_prompt(prompt_name, arguments={"topic": "MCP协议"})
        for msg in result.messages:
            if hasattr(msg.content, "text"):
                print(f"     [{msg.role}] {msg.content.text}")


# ═════════════════════════════════════════════════════════════════════════════
# 主入口：启动 MCP Client 并运行演示
# ═════════════════════════════════════════════════════════════════════════════

async def main():
    print("=" * 65)
    print("🔌 MCP Client — 连接 MCP Server + DeepSeek")
    print("=" * 65)
    print(f"  Server 路径: {SERVER_SCRIPT}")
    print(f"  LLM 模型:    {LLM_MODEL}")
    print()

    # 配置 MCP Server 的 stdio 连接参数
    # stdio 模式：Client 启动 Server 作为子进程，通过标准输入/输出通信
    server_params = StdioServerParameters(
        command=sys.executable,        # 使用当前 Python 解释器
        args=[SERVER_SCRIPT],          # Server 脚本路径
    )

    # 通过 stdio 启动 MCP Server 子进程并建立连接
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 初始化 MCP 会话（协议握手）
            await session.initialize()
            print("  ✅ MCP 连接建立成功！\n")

            # ─── 运行各演示场景 ───
            # 可以注释/取消注释来选择运行哪些场景
            await demo_basic_tool_call(session)
            await demo_calculate(session)
            await demo_stateful_tool(session)
            # await demo_multi_tool(session)       # 综合场景（耗时较长）
            await demo_resource_read(session)
            await demo_prompt_template(session)

    print("\n" + "=" * 65)
    print("✅  MCP Client 演示完毕")
    print("=" * 65)
    print()
    print("💡 学习建议：")
    print("   1. 先阅读 06_MCP_Server.py 理解工具定义方式")
    print("   2. 对比 04_Tools.py 理解 MCP 带来的的解耦优势")
    print("   3. 尝试添加新的 Tool 到 Server，观察 Client 自动发现")
    print("   4. 使用 'mcp dev 06_MCP_Server.py' 可视化调试")
    print("   5. 参考 https://modelcontextprotocol.io/ 深入学习")


if __name__ == "__main__":
    asyncio.run(main())
