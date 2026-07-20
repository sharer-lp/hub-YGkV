"""
MCP Server — 模型上下文协议 (Model Context Protocol) 服务端

MCP 是什么？   补充Streamable HTTP
  MCP (Model Context Protocol) 是 Anthropic 提出的开放协议，为 AI 模型提供了
  与外部工具、数据源交互的标准化接口。类似于 USB 协议统一了设备连接，
  MCP 统一了 AI 与工具的连接方式。

核心概念：
  - Tools:     模型可调用的函数（类似 Function Calling）
  - Resources: 模型可读取的数据源（如文件、数据库）
  - Prompts:   预定义的提示词模板

本文件实现了一个 MCP Server，提供以下能力：
  - Tool: get_current_time  — 获取当前时间
  - Tool: calculate         — 数学计算
  - Tool: manage_notes      — 简易笔记本（增/查/列/删）
  - Resource: info://server — 服务器基本信息
  - Resource: notes://all   — 所有笔记内容
  - Prompt: study_assistant — 学习助手提示词模板

运行方式：
  先安装依赖：pip install "mcp[cli]"
  然后直接运行：python 06_MCP_Server.py
  （本文件作为 MCP Server 通过 stdio 通信，由 Client 启动）

也可使用 MCP Inspector 调试：
  mcp dev 06_MCP_Server.py

协议规范：https://modelcontextprotocol.io/
"""

import json
import math
from datetime import datetime

from mcp.server.fastmcp import FastMCP

# ═════════════════════════════════════════════════════════════════════════════
# 创建 MCP Server 实例
# ═════════════════════════════════════════════════════════════════════════════

mcp = FastMCP(
    "学习助手MCP服务",
    version="1.0.0",
)

# ═════════════════════════════════════════════════════════════════════════════
# 1. Tool — 获取当前时间
# ═════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def get_current_time(timezone_offset: int = 8) -> str:
    """获取当前时间。timezone_offset 为时区偏移（默认 +8 北京时间）"""
    from datetime import timedelta, timezone
    tz = timezone(timedelta(hours=timezone_offset))
    now = datetime.now(tz)
    return json.dumps({
        "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
        "weekday": ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][now.weekday()],
        "timezone": f"UTC+{timezone_offset}",
    }, ensure_ascii=False)


# ═════════════════════════════════════════════════════════════════════════════
# 2. Tool — 数学计算器
# ═════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def calculate(expression: str) -> str:
    """
    执行数学计算，支持四则运算、幂运算和常用数学函数。

    支持的函数：sqrt, sin, cos, tan, log, log10, abs, round, pi, e
    示例：
      - "2 + 3 * 4"       → 14
      - "sqrt(144)"        → 12.0
      - "2^10"             → 1024
      - "sin(pi/2)"        → 1.0
    """
    allowed_names = {
        "abs", "round", "max", "min", "sum", "pow",
        "sqrt", "pi", "e", "sin", "cos", "tan", "log", "log10",
    }
    safe_globals = {"__builtins__": {}}
    safe_locals = {name: getattr(math, name, None) for name in allowed_names}

    expr = expression.replace("^", "**")
    try:
        result = eval(expr, safe_globals, safe_locals)
        return json.dumps({
            "expression": expression,
            "result": result,
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "expression": expression,
            "error": str(e),
        }, ensure_ascii=False)


# ═════════════════════════════════════════════════════════════════════════════
# 3. Tool — 简易笔记本（有状态的工具）
# ═════════════════════════════════════════════════════════════════════════════

_notes: list[dict] = []


@mcp.tool()
def manage_notes(action: str, content: str = "", index: int = -1) -> str:
    """
    简易笔记本管理工具。

    参数：
      - action:  操作类型，可选 "add"（添加）、"list"（列出所有）、"get"（获取指定）、"delete"（删除）
      - content: 添加笔记时的内容（action="add" 时必填）
      - index:   笔记索引（action="get" 或 "delete" 时必填，从 0 开始）

    示例：
      - manage_notes(action="add", content="今天学了MCP协议")
      - manage_notes(action="list")
      - manage_notes(action="get", index=0)
      - manage_notes(action="delete", index=0)
    """
    if action == "add":
        if not content:
            return json.dumps({"error": "content 不能为空"}, ensure_ascii=False)
        note = {
            "index": len(_notes),
            "content": content,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        _notes.append(note)
        return json.dumps({"action": "add", "note": note}, ensure_ascii=False)

    elif action == "list":
        return json.dumps({
            "action": "list",
            "total": len(_notes),
            "notes": _notes,
        }, ensure_ascii=False)

    elif action == "get":
        if 0 <= index < len(_notes):
            return json.dumps({"action": "get", "note": _notes[index]}, ensure_ascii=False)
        return json.dumps({"error": f"索引 {index} 超出范围 (共 {len(_notes)} 条)"}, ensure_ascii=False)

    elif action == "delete":
        if 0 <= index < len(_notes):
            removed = _notes.pop(index)
            # 重新编号
            for i, n in enumerate(_notes):
                n["index"] = i
            return json.dumps({"action": "delete", "removed": removed}, ensure_ascii=False)
        return json.dumps({"error": f"索引 {index} 超出范围 (共 {len(_notes)} 条)"}, ensure_ascii=False)

    else:
        return json.dumps({"error": f"未知操作: {action}，支持: add/list/get/delete"}, ensure_ascii=False)


# ═════════════════════════════════════════════════════════════════════════════
# 4. Resource — 暴露静态资源（模型可读取的数据）
# ═════════════════════════════════════════════════════════════════════════════

@mcp.resource("info://server")
def server_info() -> str:
    """返回 MCP Server 的基本信息"""
    return json.dumps({
        "name": "学习助手MCP服务",
        "version": "1.0.0",
        "tools": ["get_current_time", "calculate", "manage_notes"],
        "description": "这是一个用于学习 MCP 协议的示例服务器",
    }, ensure_ascii=False)


@mcp.resource("notes://all")
def all_notes() -> str:
    """返回所有笔记内容（作为 Resource 暴露）"""
    return json.dumps({
        "total": len(_notes),
        "notes": _notes,
    }, ensure_ascii=False)


# ═════════════════════════════════════════════════════════════════════════════
# 5. Prompt — 预定义提示词模板
# ═════════════════════════════════════════════════════════════════════════════

@mcp.prompt()
def study_assistant(topic: str) -> str:
    """生成一个学习助手的系统提示词"""
    return f"""你是一个专业的学习助手。用户正在学习「{topic}」相关内容。

请注意：
1. 用通俗易懂的语言解释概念
2. 多举生活中的例子
3. 如果涉及计算，请使用 calculate 工具确保准确性
4. 可以记录用户的学习笔记

请问用户有什么具体问题？"""


# ═════════════════════════════════════════════════════════════════════════════
# 启动 MCP Server
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🚀 MCP Server 启动中... (通过 stdio 通信)")
    print("   请使用 Client 端启动本服务，或运行: mcp dev 06_MCP_Server.py")
    mcp.run(transport="stdio")
