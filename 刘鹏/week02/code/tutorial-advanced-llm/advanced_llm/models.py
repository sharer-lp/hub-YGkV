"""
数据模型定义 — 全项目统一的"数据容器"

==================== 设计说明 ====================

【本文件职责】
    将所有数据结构集中管理，与业务逻辑分离。
    这是"关注点分离"（Separation of Concerns）原则的体现。

【为什么要单独一个 models.py？】
    1. 避免循环导入：
       - clients/openai_client.py 需要 ChatResult
       - clients/anthropic_client.py 也需要 ChatResult
       - factory.py 需要引用 clients
       如果 ChatResult 定义在某个 client 里，就会产生 A→B→A 循环依赖。
       抽到独立的 models.py → 所有人都依赖它，它不依赖任何人。

    2. 修改集中：
       新增响应字段只改这一个文件，不用到处找。

    3. IDE 友好：
       类型定义清晰，自动补全和类型检查都能正常工作。

【包含的数据结构】
    - FinishReason      结束原因枚举（stop / length / tool_calls）
    - ChatResult        非流式调用统一返回
    - StreamChunk       流式调用单个 chunk
    - ToolCallRequest   工具调用请求（模型返回的）
    - ToolCallResult    工具调用结果（我们执行后返回的）
    - ConversationTurn  单轮对话记录（含思考内容）

【核心知识点：@dataclass vs 普通 class】
    普通写法（冗余）：
        class ChatResult:
            def __init__(self, content=None, reasoning=None):
                self.content = content
                self.reasoning = reasoning
            def __repr__(self):
                return f"ChatResult(content={self.content}, ...)"
            def __eq__(self, other):
                return self.content == other.content and ...

    dataclass 写法（一行搞定）：
        @dataclass
        class ChatResult:
            content: Optional[str] = None
            reasoning: Optional[str] = None

    dataclass 自动生成 __init__、__repr__、__eq__，代码量减少 70%。

【核心知识点：Optional 类型注解】
    Optional[str] 等价于 str | None，表示"可能是字符串，也可能是 None"。
    这是 Python 类型提示（Type Hints）的一部分：
        - 不影响运行（Python 不强制类型检查）
        - 但 IDE 能据此提供智能提示和错误检测
        - mypy / pyright 等工具可以做静态类型检查

==================================================
"""

# ==================== 导入区 ====================
from dataclasses import dataclass, field   # 数据类装饰器
from typing import Optional, Any           # 类型注解工具
from enum import Enum                      # 枚举类型


# ==================== 枚举类型 ====================
# 【知识点：Enum 枚举】
#   枚举用于定义"一组固定的常量"，比用字符串或数字更安全：
#     - 防止拼写错误：FinishReason.STOP 写错会报 AttributeError
#     - 如果用字符串 "stop" 写错成 "stpo" 不会报错，只会产生隐蔽 bug
#     - IDE 能自动补全所有可选值
#
#   继承 str 的好处（str, Enum）：
#     - FinishReason.STOP == "stop"  → True（可以直接和字符串比较）
#     - JSON 序列化时自动变成字符串
#     - 如果不继承 str，则需要 .value 才能取到字符串值

class FinishReason(str, Enum):
    """模型响应结束原因枚举

    【各值含义】
    - STOP:           正常结束（模型说完了）
    - LENGTH:         达到 max_tokens 截断（模型还没说完但被强制停止）
    - TOOL_CALLS:     模型请求调用工具（还没回答完，需要先执行工具）
    - CONTENT_FILTER: 安全审查拦截（触发了内容安全策略）

    【生产中的用途】
    - finish_reason == "tool_calls" → 进入工具调用循环
    - finish_reason == "length"    → 提示用户"回答被截断"或增大 max_tokens
    - finish_reason == "stop"      → 正常展示回答
    """
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"


# ==================== 非流式调用结果 ====================
# 【知识点：dataclass 中的默认值】
#   字段定义顺序：无默认值的字段必须在有默认值的字段前面。
#   所有字段都给 None 默认值 → 可以 ChatResult() 创建空对象，再逐步填充。
#
# 【知识点：field(default_factory=list)】
#   可变默认值（list/dict）不能用 = []，否则所有实例共享同一个列表！
#   必须用 field(default_factory=list) 让每个实例创建独立的列表。
#   本项目中用 Optional[list] = None 替代，避免这个问题。

@dataclass
class ChatResult:
    """非流式调用统一返回结构

    【设计思想：统一返回格式】
    不同厂商的 API 返回结构各不相同：
        - OpenAI:    response.choices[0].message.content
        - Anthropic: response.content[0].text
    我们在客户端层将它们统一转换为 ChatResult，
    上层业务代码只需要处理一种格式 → 这就是"适配器模式"的价值。

    Attributes:
        content:       模型回答文本（最终给用户看的内容）
        reasoning:     思考过程（开启 thinking 时才有，类似"草稿纸"）
        usage:         token 消耗统计（用于计费监控）
        finish_reason: 结束原因（stop / length / tool_calls 等）
        tool_calls:    工具调用请求列表（finish_reason=tool_calls 时有值）
        raw_message:   原始 message 对象（用于多轮对话拼接，保留完整信息）
    """
    content: Optional[str] = None
    reasoning: Optional[str] = None
    usage: Optional[dict] = None
    finish_reason: Optional[str] = None
    tool_calls: Optional[list["ToolCallRequest"]] = None
    raw_message: Any = None  # 保留原始对象，用于工具调用场景拼接

    # ---- 便捷属性（@property 将方法伪装成属性访问） ----

    @property
    def has_tool_calls(self) -> bool:
        """是否包含工具调用请求

        【知识点：bool() 的隐式转换】
        bool(None) → False
        bool([])   → False（空列表）
        bool([x])  → True（非空列表）
        所以 `bool(self.tool_calls)` 能同时处理 None 和空列表两种情况。
        """
        return bool(self.tool_calls)

    @property
    def is_truncated(self) -> bool:
        """是否被 max_tokens 截断（回答不完整）"""
        return self.finish_reason == FinishReason.LENGTH

    def to_dict(self) -> dict:
        """转为字典（方便 JSON 序列化、日志记录、API 响应）

        【知识点：为什么不直接 vars(self) 或 __dict__？】
        因为 dataclass 的 __dict__ 包含 raw_message（不可序列化的对象），
        手动选择字段可以：
            1. 排除不可序列化的字段
            2. 添加计算属性（如 has_tool_calls）
            3. 控制对外暴露的字段（封装原则）
        """
        return {
            "content": self.content,
            "reasoning": self.reasoning,
            "usage": self.usage,
            "finish_reason": self.finish_reason,
            "has_tool_calls": self.has_tool_calls,
        }


# ==================== 流式调用 chunk ====================
# 【知识点：流式输出（Streaming）】
#   普通调用：等模型全部生成完 → 一次性返回（用户等待时间长）
#   流式调用：模型每生成几个字 → 立即推送给客户端（打字机效果）
#
#   技术实现：SSE（Server-Sent Events）协议
#     - HTTP 响应不关闭，持续发送数据块
#     - 每个数据块就是一个 StreamChunk
#     - 客户端用 for 循环逐个接收
#
#   chunk.type 的生命周期：
#     reasoning → reasoning → ... → content → content → ... → done
#     （先思考，再回答，最后结束）

@dataclass
class StreamChunk:
    """流式调用单个数据块

    Attributes:
        type:  chunk 类型 → "reasoning" | "content" | "tool_call" | "done"
        data:  增量文本（注意是"增量"而非全量，需要客户端自己拼接）
        usage: token 统计（仅 done 阶段有值）
    """
    type: str       # "reasoning" | "content" | "tool_call" | "done"
    data: Optional[str] = None
    usage: Optional[dict] = None

    def to_dict(self) -> dict:
        return {"type": self.type, "data": self.data, "usage": self.usage}


# ==================== 工具调用相关 ====================
# 【知识点：Function Calling / Tool Calls 的本质】
#   AI 模型不能真正"执行"任何代码！它只是"说"要调什么函数、传什么参数。
#   真正执行函数的是你的代码（tool_registry.py）。
#
#   完整流程：
#     1. 你告诉模型有哪些工具可用（tools 参数）
#     2. 模型判断需要调用工具 → 返回 tool_calls（函数名 + 参数）
#     3. 你的代码解析参数 → 执行函数 → 获取结果
#     4. 将结果以 role="tool" 消息喂回模型
#     5. 模型基于工具结果生成最终回答
#
# 【关键细节】
#   - arguments 是 JSON 字符串（不是字典！）需要 json.loads() 解析
#   - tool_call_id 必须一一对应（模型靠它匹配"哪个结果对应哪个调用"）

@dataclass
class ToolCallRequest:
    """工具调用请求（模型返回的"我想调用某个函数"）

    Attributes:
        id:        调用唯一标识（喂回结果时必须对应，类似"订单号"）
        name:      函数名（如 "get_weather"）
        arguments: 参数字符串（JSON 格式，如 '{"city": "北京"}'）
                   注意：是字符串不是字典！需要 json.loads() 解析
    """
    id: str
    name: str
    arguments: str  # JSON 字符串，不是 dict！

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}


@dataclass
class ToolCallResult:
    """工具调用执行结果（我们的代码执行函数后的返回值）

    【设计思想：结构化错误返回】
    工具执行可能失败（网络超时、参数错误等），
    但我们不能直接抛异常让程序崩溃。
    正确做法：将错误信息包装为结构化结果，喂回给模型，
    让模型自己决定如何处理（重试、换工具、或告知用户）。

    Attributes:
        tool_call_id: 对应的调用 ID（必须与 ToolCallRequest.id 一致）
        name:         函数名
        result:       执行结果字符串（成功时是正常结果，失败时是错误信息）
        success:      是否执行成功
        error:        错误信息（失败时有值）
    """
    tool_call_id: str
    name: str
    result: str
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "result": self.result,
            "success": self.success,
            "error": self.error,
        }


# ==================== 多轮对话记录 ====================
# 【核心规则（DeepSeek 官方文档明确要求）】
#   ┌─────────────────────────────────────────────────────────────────┐
#   │ 无工具调用时：reasoning_content 无需回传（可丢弃，节省 token） │
#   │ 有工具调用时：reasoning_content 必须回传（否则 API 报错）     │
#   └─────────────────────────────────────────────────────────────────┘
#   这个规则在 build_context_messages() 中自动处理，调用方无需关心。

@dataclass
class ConversationTurn:
    """单轮对话完整记录（含思考内容和工具调用）

    【设计思想】
    一轮对话可能包含多条消息（特别是工具调用场景）：
        user → assistant(含 tool_calls) → tool(结果1) → tool(结果2) → assistant(最终回答)
    ConversationTurn 将这一整轮打包为一个对象，方便管理和序列化。

    Attributes:
        user_message:      用户消息
        assistant_content: 助手最终回答
        reasoning:         思考过程（草稿纸内容）
        tool_calls:        工具调用请求列表
        tool_results:      工具执行结果列表
        usage:             token 消耗
        has_tool_calls:    本轮是否涉及工具调用（决定 reasoning 是否必须保留）
    """
    user_message: str
    assistant_content: Optional[str] = None
    reasoning: Optional[str] = None
    tool_calls: Optional[list[ToolCallRequest]] = None
    tool_results: Optional[list[ToolCallResult]] = None
    usage: Optional[dict] = None
    has_tool_calls: bool = False

    def build_context_messages(self, include_reasoning: bool = False) -> list[dict]:
        """
        构建用于 API 调用的 messages 列表

        关键规则（DeepSeek 官方文档）：
        - 无工具调用时：reasoning_content 无需回传，只保留 content
        - 有工具调用时：reasoning_content 必须回传，否则上下文不连贯

        Args:
            include_reasoning: 是否强制包含 reasoning（覆盖自动判断）

        Returns:
            messages 列表
        """
        messages = []

        # 判断是否需要包含 reasoning
        should_include_reasoning = include_reasoning or self.has_tool_calls

        if self.has_tool_calls and self.tool_calls:
            # 有工具调用的场景：必须完整保留 reasoning + tool_calls + tool results
            assistant_msg = {
                "role": "assistant",
                "content": self.assistant_content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": tc.arguments},
                    }
                    for tc in self.tool_calls
                ],
            }
            # 有工具调用时，reasoning 必须包含
            if self.reasoning and should_include_reasoning:
                assistant_msg["reasoning_content"] = self.reasoning

            messages.append(assistant_msg)

            # 添加工具执行结果
            if self.tool_results:
                for tr in self.tool_results:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tr.tool_call_id,
                        "content": tr.result,
                    })
        else:
            # 无工具调用：只保留 content（reasoning 可选）
            assistant_msg = {
                "role": "assistant",
                "content": self.assistant_content,
            }
            if self.reasoning and should_include_reasoning:
                assistant_msg["reasoning_content"] = self.reasoning

            messages.append(assistant_msg)

        return messages
