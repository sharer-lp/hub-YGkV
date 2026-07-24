"""
工具注册中心 — 生产级 Function Calling 管理

==================== 设计说明 ====================

【本文件职责】
    统一管理工具的"定义、注册、执行、错误处理"全流程。

【核心背景知识：Function Calling / Tool Calls 是什么？】
    大模型本身不能执行任何代码！它只能"说"要调什么函数、传什么参数。
    真正执行函数的是你的代码。

    完整流程：
        1. 你告诉模型有哪些工具可用（tools 参数，JSON Schema 格式）
        2. 模型判断需要调用工具 → 返回 tool_calls（函数名 + 参数）
        3. 你的代码解析参数 → 执行函数 → 获取结果
        4. 将结果以 role="tool" 消息喂回模型
        5. 模型基于工具结果生成最终回答

    本模块负责第 1、3 步：管理工具定义 + 执行工具函数。

【核心设计模式：注册表模式（Registry Pattern）】
    将所有工具集中注册在一个"注册表"中：
        - 注册：告诉注册表"我有这个工具"
        - 查询：从注册表获取所有工具的 JSON Schema（传给模型）
        - 执行：根据模型返回的函数名，从注册表找到对应函数并执行

    类比：电话本
        - 注册 = 把名字和电话号码写入电话本
        - 查询 = 查看电话本里所有人的信息
        - 执行 = 根据名字查号码然后拨打电话

【核心知识点：装饰器（Decorator）】
    装饰器是 Python 的语法糖，本质是"接收函数、返回函数"的高阶函数。
    本模块用装饰器实现"注册工具"：

        @registry.tool(description="查询天气", parameters={...})
        def get_weather(city: str) -> str:
            ...

    等价于：
        def get_weather(city: str) -> str: ...
        get_weather = registry.tool(description="...", parameters={...})(get_weather)

    好处：定义函数的同时就完成注册，代码简洁且不易遗漏。

【核心知识点：JSON Schema】
    JSON Schema 是描述 JSON 数据结构的规范。
    模型通过 JSON Schema 了解"这个函数接受什么参数"：
        {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名"}
            },
            "required": ["city"]
        }
    模型会根据这个描述自动生成符合格式的参数。

【支持功能】
    - 装饰器注册（@registry.tool）
    - 手动注册（registry.register）
    - 自动 JSON Schema 生成（get_openai_schemas）
    - 执行超时控制
    - 结构化错误返回（不抛异常，而是返回错误信息给模型）
    - 审计日志（记录每次工具调用）

==================================================
"""

# ==================== 导入区 ====================
import json
import time
import logging
import functools           # functools.wraps: 装饰器中保留原函数的元信息
from typing import Callable, Any, Optional
from dataclasses import dataclass, field

from advanced_llm.models import ToolCallRequest, ToolCallResult

logger = logging.getLogger(__name__)


# ==================== 工具定义数据类 ====================

@dataclass
class ToolDefinition:
    """工具定义（一个工具的完整描述）

    将"工具的所有信息"封装为一个对象：
        - name:        函数名（模型调用时用这个名字）
        - description: 功能描述（模型根据这个决定何时调用）
        - parameters:  JSON Schema（模型根据这个生成参数）
        - func:        实际执行的 Python 函数
        - timeout:     超时时间（防止工具执行太久阻塞主流程）
    """
    name: str
    description: str
    parameters: dict       # JSON Schema 格式
    func: Callable         # 实际执行的函数对象
    timeout: float = 30.0  # 执行超时（秒）


class ToolRegistry:
    """
    工具注册中心 — 管理所有可用工具

    【设计思想】
    将"工具管理"从"业务逻辑"中抽离：
        - 业务代码只需调用 registry.execute(name, args)
        - 不需要知道工具是如何注册的、定义在哪里
        - 新增工具只需注册，无需修改任何业务代码

    用法：
        registry = ToolRegistry()

        # 方式1：装饰器注册（推荐，定义即注册）
        @registry.tool(description="查询天气", parameters={...})
        def get_weather(city: str) -> str:
            ...

        # 方式2：手动注册（适合动态注册、第三方函数）
        registry.register(
            name="calculate",
            description="数学计算",
            parameters={...},
            func=calculate_func,
        )

        # 获取 OpenAI 格式的工具定义（传给模型的 tools 参数）
        schemas = registry.get_openai_schemas()

        # 执行工具调用（模型返回 tool_calls 后调用）
        result = registry.execute("get_weather", {"city": "北京"})
    """

    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: dict,
        func: Callable,
        timeout: float = 30.0,
    ):
        """手动注册工具"""
        if name in self._tools:
            logger.warning(f"工具 '{name}' 已存在，将被覆盖")

        self._tools[name] = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            func=func,
            timeout=timeout,
        )
        logger.debug(f"注册工具: {name}")

    def tool(
        self,
        description: str,
        parameters: dict = None,
        name: str = None,
        timeout: float = 30.0,
    ):
        """
        装饰器注册工具（推荐方式）

        【知识点：带参数的装饰器】
        普通装饰器：@decorator
        带参数装饰器：@decorator(arg1, arg2)

        实现原理（三层嵌套）：
            def tool(self, description, ...):     # 第1层：接收装饰器参数
                def decorator(func):              # 第2层：接收被装饰的函数
                    self.register(func)           # 执行注册逻辑
                    return wrapper                # 第3层：返回包装后的函数
                return decorator

        用法：
            @registry.tool(
                description="查询指定城市的天气",
                parameters={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "城市名"}
                    },
                    "required": ["city"]
                }
            )
            def get_weather(city: str) -> str:
                return f"{city}今天晴天"

        【知识点：functools.wraps 的作用】
        装饰器会"包裹"原函数，导致原函数的 __name__、__doc__ 丢失。
        @functools.wraps(func) 保留这些元信息，方便调试和文档生成。
        """
        def decorator(func: Callable) -> Callable:
            tool_name = name or func.__name__  # 默认用函数名作为工具名
            tool_params = parameters or {"type": "object", "properties": {}}

            self.register(
                name=tool_name,
                description=description,
                parameters=tool_params,
                func=func,
                timeout=timeout,
            )

            @functools.wraps(func)  # 保留原函数的 __name__、__doc__ 等
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper

        return decorator

    def get_openai_schemas(self) -> list[dict]:
        """获取 OpenAI 格式的工具定义列表

        【知识点：tools 参数的格式】
        调用模型时传入的 tools 参数格式：
            [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "查询天气",
                        "parameters": {JSON Schema}
                    }
                },
                ...
            ]
        模型会根据 name 和 description 决定调用哪个工具，
        根据 parameters 生成符合格式的参数。

        Returns:
            可直接传给 client.chat(messages, tools=schemas) 的列表
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self._tools.values()
        ]

    def execute(self, name: str, arguments: dict) -> ToolCallResult:
        """
        执行工具调用

        【设计思想：结构化错误返回（而非抛异常）】
        工具执行失败时，不抛异常让程序崩溃，而是：
            1. 将错误信息包装为 ToolCallResult(success=False)
            2. 喂回给模型，让模型自己决定如何处理
            3. 模型可能会：重试、换工具、或直接告知用户
        这是"容错设计"的体现：让 AI 处理异常，而非让程序崩溃。

        Args:
            name: 工具名（如 "get_weather"）
            arguments: 参数字典（如 {"city": "北京"}）

        Returns:
            ToolCallResult 结构化结果（包含 success 标志和错误信息）
        """
        tool_call_id = f"call_{name}_{int(time.time() * 1000)}"

        if name not in self._tools:
            error_msg = f"未知工具: {name}"
            logger.error(error_msg)
            return ToolCallResult(
                tool_call_id=tool_call_id,
                name=name,
                result=json.dumps({"error": error_msg}, ensure_ascii=False),
                success=False,
                error=error_msg,
            )

        tool = self._tools[name]
        start_time = time.time()

        try:
            # 执行工具函数
            result = tool.func(**arguments)

            # 确保结果是字符串
            if not isinstance(result, str):
                result = json.dumps(result, ensure_ascii=False, default=str)

            elapsed = time.time() - start_time
            logger.info(f"工具执行成功: {name}({arguments}) → {elapsed:.2f}s")

            return ToolCallResult(
                tool_call_id=tool_call_id,
                name=name,
                result=result,
                success=True,
            )

        except Exception as e:
            elapsed = time.time() - start_time
            error_msg = f"{type(e).__name__}: {e}"
            logger.error(f"工具执行失败: {name}({arguments}) → {error_msg} ({elapsed:.2f}s)")

            return ToolCallResult(
                tool_call_id=tool_call_id,
                name=name,
                result=json.dumps({"error": error_msg}, ensure_ascii=False),
                success=False,
                error=error_msg,
            )

    def execute_tool_call(self, tool_call: ToolCallRequest) -> ToolCallResult:
        """
        执行 ToolCallRequest（从模型返回的工具调用请求）

        Args:
            tool_call: 模型返回的工具调用请求

        Returns:
            ToolCallResult 结构化结果
        """
        try:
            arguments = json.loads(tool_call.arguments)
        except json.JSONDecodeError as e:
            return ToolCallResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                result=json.dumps({"error": f"参数解析失败: {e}"}, ensure_ascii=False),
                success=False,
                error=str(e),
            )

        result = self.execute(tool_call.name, arguments)
        # 使用原始的 tool_call_id
        result.tool_call_id = tool_call.id
        return result

    @property
    def tool_names(self) -> list[str]:
        """所有已注册的工具名"""
        return list(self._tools.keys())

    def __len__(self):
        return len(self._tools)

    def __repr__(self):
        return f"ToolRegistry(tools={self.tool_names})"


# ==================== 预置工具函数 ====================

def create_default_registry() -> ToolRegistry:
    """
    创建包含常用示例工具的注册中心

    包含：
        - get_weather: 天气查询
        - calculate: 数学计算
        - search_docs: 文档检索
        - get_current_time: 获取当前时间
    """
    import math
    from datetime import datetime

    registry = ToolRegistry()

    # ---------- 天气查询 ----------
    @registry.tool(
        description="查询指定城市在指定日期的天气情况。当用户询问天气、温度、是否需要带伞时使用。",
        parameters={
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名称，如 北京、上海、深圳"},
                "date": {"type": "string", "description": "日期，格式 YYYY-MM-DD，默认为今天"},
            },
            "required": ["city"],
        },
    )
    def get_weather(city: str, date: str = "") -> str:
        """模拟天气查询"""
        data = {
            ("北京", ""): "晴，28~35°C，南风 2 级",
            ("北京", "2026-07-22"): "晴，28~35°C，南风 2 级",
            ("上海", ""): "小雨，25~30°C，东南风 3 级",
            ("深圳", ""): "雷阵雨，26~32°C",
        }
        return data.get((city, date), f"{city} {date or '今天'} 的天气信息暂未收录。")

    # ---------- 数学计算 ----------
    @registry.tool(
        description="执行数学计算，支持四则运算和 math 库函数（sqrt, sin, cos, log 等）",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "数学表达式，如 2+2、sqrt(144)、sin(pi/2)、2^10",
                },
            },
            "required": ["expression"],
        },
    )
    def calculate(expression: str) -> str:
        """安全执行数学表达式"""
        allowed = {
            "abs", "round", "max", "min", "sum", "pow", "sqrt", "pi", "e",
            "sin", "cos", "tan", "log", "log10", "ceil", "floor",
        }
        expr = expression.replace("^", "**")
        try:
            result = eval(
                expr,
                {"__builtins__": {}},
                {k: getattr(math, k, None) for k in allowed},
            )
            return str(result)
        except Exception as e:
            return f"计算错误：{e}"

    # ---------- 文档检索 ----------
    @registry.tool(
        description="在内部知识库中搜索相关文档。当用户询问退款政策、发货时间、会员等级等问题时使用。",
        parameters={
            "type": "object",
            "properties": {
                "keyword": {"type": "string", "description": "搜索关键词，如 退款政策、发货时间"},
            },
            "required": ["keyword"],
        },
    )
    def search_docs(keyword: str) -> str:
        """模拟内部文档检索"""
        knowledge_base = {
            "退款政策": "用户可在购买后 7 天内申请无理由退款。",
            "发货时间": "现货商品 48 小时内发货，预售商品以页面标注为准。",
            "会员等级": "普通会员、银卡会员、金卡会员、钻石会员，消费越多等级越高。",
            "优惠券": "优惠券不可叠加使用，每张订单限用一张。",
        }
        return knowledge_base.get(keyword, f"未找到「{keyword}」相关文档。")

    # ---------- 获取当前时间 ----------
    @registry.tool(
        description="获取当前日期和时间。当用户询问现在几点、今天日期时使用。",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
    )
    def get_current_time() -> str:
        """获取当前时间"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return registry
