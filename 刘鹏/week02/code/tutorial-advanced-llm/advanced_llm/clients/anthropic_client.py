"""
Anthropic 格式 LLM 客户端 — Claude 系列模型专用

==================== 适用模型 ====================

Anthropic Claude 系列：
    - claude-3-5-sonnet / claude-3-opus / claude-3-haiku
    - claude-sonnet-4-20250514 等后续版本

【核心知识点：Anthropic 与 OpenAI 的协议差异】
    这是本文件存在的根本原因。两者虽然都是"对话 API"，但格式差异显著：

    ┌─────────────────┬──────────────────────────┬──────────────────────────────┐
    │ 差异点          │ OpenAI 格式              │ Anthropic 格式               │
    ├─────────────────┼──────────────────────────┼──────────────────────────────┤
    │ system 消息     │ 放在 messages 数组中     │ 独立的 system 参数           │
    │ 工具定义        │ function.parameters      │ input_schema                 │
    │ 工具调用响应    │ message.tool_calls 列表  │ content 中的 tool_use block  │
    │ 工具结果回传    │ role="tool"              │ role="user" + tool_result    │
    │ 思考模式        │ extra_body / reasoning   │ thinking={budget_tokens}     │
    │ 响应结构        │ choices[0].message       │ content[] blocks             │
    │ 结束原因字段    │ finish_reason            │ stop_reason                  │
    │ token 统计      │ prompt/completion_tokens │ input/output_tokens          │
    └─────────────────┴──────────────────────────┴──────────────────────────────┘

    本客户端的职责：对外暴露与 OpenAIClient 相同的接口（chat/chat_stream），
    内部自动完成 OpenAI → Anthropic 的格式转换。上层代码完全无感知。
    这就是"适配器模式"（Adapter Pattern）的经典应用。

【协议特点】
    - 使用 anthropic SDK（pip install anthropic）
    - 思考模式通过 thinking={"type": "enabled", "budget_tokens": N} 控制
    - budget_tokens 控制思考预算（思考最多用多少 token）
    - 消息格式与 OpenAI 不同（system 单独传递，不在 messages 中）
    - 响应是 content blocks 列表（而非单一 message 对象）

==================================================
"""

# ==================== 导入区 ====================
import json
import logging
from typing import Optional, Generator

from advanced_llm.config import cfg
from advanced_llm.models import ChatResult, StreamChunk, ToolCallRequest
from advanced_llm.clients.base import BaseLLMClient

logger = logging.getLogger(__name__)


class AnthropicClient(BaseLLMClient):
    """
    Anthropic 格式客户端（Claude 系列）

    【继承关系】
        BaseLLMClient (ABC)
        └── AnthropicClient  ← 你在这里

    【适配器模式体现】
        对外接口：与 OpenAIClient 完全相同（chat / chat_stream）
        内部实现：自动将 OpenAI 格式转换为 Anthropic 格式
        上层代码：无需知道底层是 Anthropic，写法与调用 OpenAI 完全一样

    用法：
        client = AnthropicClient("claude")
        result = client.chat(messages)         # 内部自动转换格式
        for chunk in client.chat_stream(messages): ...
    """

    def __init__(self, model_name: str = None):
        """
        初始化 Anthropic 客户端

        【知识点：延迟导入 + 友好错误提示】
        anthropic SDK 是可选依赖（不是所有人都用 Claude）。
        如果放在文件顶部 import，未安装时整个模块都无法加载。
        放在 __init__ 内部：只有真正创建 AnthropicClient 时才检查，
        并给出清晰的安装指引（而非让用户看 ImportError traceback）。
        """
        super().__init__(model_name)

        try:
            import anthropic  # 延迟导入：只有用到 Claude 时才需要这个包
        except ImportError as e:
            raise ImportError(
                "使用 Anthropic 客户端需要安装 anthropic SDK\n"
                "请执行: pip install anthropic"
            ) from e  # from e 保留原始异常链

        model_cfg = cfg.get_model(model_name) if model_name else cfg.active_model

        # 创建 Anthropic SDK 客户端实例
        self.client = anthropic.Anthropic(
            api_key=model_cfg.api_key,
            base_url=model_cfg.base_url or None,  # None 时使用 SDK 默认地址
            timeout=cfg.TIMEOUT,
            max_retries=cfg.MAX_RETRIES,
        )

    # ==================== 消息格式转换（适配器核心） ====================
    # 【设计思想】
    #   上层代码统一使用 OpenAI 格式的 messages：
    #     [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
    #   但 Anthropic API 要求：
    #     - system 是独立参数：client.messages.create(system="...", messages=[...])
    #     - messages 中不能有 role="system"
    #     - 工具结果用 role="user" + tool_result block（而非 role="tool"）
    #   所以需要一个"翻译层"将 OpenAI 格式转为 Anthropic 格式。

    @staticmethod
    def _split_system_and_messages(messages: list[dict]) -> tuple[str, list[dict]]:
        """
        将 OpenAI 格式的 messages 转换为 Anthropic 格式

        【转换规则】
        1. role="system" → 提取为独立的 system 字符串（多个 system 用换行拼接）
        2. role="tool"   → 转为 role="user" + tool_result content block
        3. role="assistant" + tool_calls → 转为 tool_use content blocks
        4. 其他消息 → 保持原样

        【知识点：tuple 返回值】
        函数返回 tuple[str, list[dict]]，调用方用解构赋值接收：
            system, converted_messages = self._split_system_and_messages(messages)
        这是 Python 中"返回多个值"的惯用方式。

        Args:
            messages: OpenAI 格式的完整 messages 列表

        Returns:
            (system_text, anthropic_messages) 元组
        """
        system_parts = []
        converted = []

        for msg in messages:
            if msg["role"] == "system":
                # system 消息提取出来，后续作为独立参数传递
                system_parts.append(msg["content"])

            elif msg["role"] == "tool":
                # 【格式差异】OpenAI 用 role="tool"，Anthropic 用 role="user" + tool_result
                converted.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.get("tool_call_id", ""),
                        "content": msg["content"],
                    }],
                })

            elif msg["role"] == "assistant" and msg.get("tool_calls"):
                # 【格式差异】OpenAI 用 tool_calls 数组，Anthropic 用 content blocks
                content = []
                if msg.get("content"):
                    content.append({"type": "text", "text": msg["content"]})
                for tc in msg["tool_calls"]:
                    # 兼容 dict 和对象两种格式（防御性编程）
                    func = tc["function"] if isinstance(tc, dict) else tc.function
                    args = func["arguments"] if isinstance(func, dict) else func.arguments
                    if isinstance(args, str):
                        args = json.loads(args)  # Anthropic 要求 input 是 dict，不是 JSON 字符串
                    content.append({
                        "type": "tool_use",
                        "id": tc["id"] if isinstance(tc, dict) else tc.id,
                        "name": func["name"] if isinstance(func, dict) else func.name,
                        "input": args,
                    })
                converted.append({"role": "assistant", "content": content})

            else:
                # 普通 user/assistant 消息，格式相同，直接保留
                converted.append({"role": msg["role"], "content": msg["content"]})

        system = "\n\n".join(system_parts)
        return system, converted

    @staticmethod
    def _convert_tools(tools: list[dict]) -> list[dict]:
        """将 OpenAI 格式的工具定义转换为 Anthropic 格式

        【格式对比】
        OpenAI 格式：
            {"type": "function", "function": {"name": "...", "parameters": {...}}}
        Anthropic 格式：
            {"name": "...", "description": "...", "input_schema": {...}}

        关键差异：
            - OpenAI 用 "parameters"，Anthropic 用 "input_schema"
            - OpenAI 多一层 "type": "function" 包装
            - Anthropic 的 name/description 在顶层
        """
        anthropic_tools = []
        for tool in tools:
            func = tool["function"]
            anthropic_tools.append({
                "name": func["name"],
                "description": func.get("description", ""),
                "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
            })
        return anthropic_tools

    # ==================== 请求参数构建 ====================

    def _build_kwargs(
        self,
        stream: bool = False,
        enable_thinking: bool = None,
        temperature: float = None,
        max_tokens: int = None,
        tools: list = None,
        tool_choice: str = None,
        **extra_kwargs,
    ) -> dict:
        """构建 Anthropic 请求参数

        【知识点：Anthropic 思考模式 vs OpenAI/DeepSeek】
        - DeepSeek:  extra_body={"thinking": {"type": "enabled"}}  （开关式）
        - Anthropic: thinking={"type": "enabled", "budget_tokens": N} （预算式）
        - OpenAI:    reasoning_effort="high"  （强度式）

        budget_tokens 的含义：
            模型思考过程最多使用 N 个 token。
            例如 budget_tokens=4096 → 思考最多用 4096 token，
            剩余 max_tokens - 4096 留给最终回答。

        【重要限制】
        思考模式下不能设置 temperature（与 DeepSeek 相同的限制）。
        Anthropic 官方文档：thinking 和 temperature 互斥。
        """
        thinking = enable_thinking if enable_thinking is not None else self.enable_thinking

        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens or cfg.MAX_TOKENS,
        }

        # Anthropic 思考模式：thinking={"type": "enabled", "budget_tokens": N}
        if thinking and self.supports_thinking:
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": cfg.THINKING_BUDGET_TOKENS,
            }
        else:
            # 非思考模式下可设置 temperature（思考模式下禁止）
            kwargs["temperature"] = temperature if temperature is not None else cfg.TEMPERATURE

        # 工具调用（需要格式转换）
        if tools:
            kwargs["tools"] = self._convert_tools(tools)
            # 【格式差异】tool_choice 的表示方式不同
            # OpenAI: "auto" / "none" / "required" / {"type":"function","function":{"name":"xxx"}}
            # Anthropic: {"type":"auto"} / {"type":"none"} / {"type":"any"} / {"type":"tool","name":"xxx"}
            if tool_choice == "none":
                kwargs["tool_choice"] = {"type": "none"}
            elif tool_choice == "required":
                kwargs["tool_choice"] = {"type": "any"}  # OpenAI 的 required = Anthropic 的 any
            elif tool_choice and tool_choice != "auto":
                kwargs["tool_choice"] = {"type": "tool", "name": tool_choice}

        kwargs.update(extra_kwargs)
        return kwargs

    # ==================== 非流式调用 ====================

    def chat(self, messages: list[dict], **kwargs) -> ChatResult:
        """非流式调用

        【调用流程】
            1. 从 kwargs 中提取已知参数（pop 出来避免重复传递）
            2. 将 OpenAI 格式 messages 转换为 Anthropic 格式
            3. 构建请求参数
            4. 调用 Anthropic API
            5. 解析 content blocks → 统一 ChatResult

        【知识点：kwargs.pop() 的妙用】
        pop(key, default) 从字典中取出并删除指定键。
        为什么用 pop 而非 get？
            - 取出后剩余的 kwargs 只包含"未知参数"，可以安全透传
            - 避免同一个参数被传递两次导致冲突
        """
        # 提取已知参数（剩余的 kwargs 是额外透传参数）
        tools = kwargs.pop("tools", None)
        tool_choice = kwargs.pop("tool_choice", None)
        enable_thinking = kwargs.pop("enable_thinking", None)
        temperature = kwargs.pop("temperature", None)
        max_tokens = kwargs.pop("max_tokens", None)

        # 格式转换：OpenAI → Anthropic
        system, converted_messages = self._split_system_and_messages(messages)

        request_kwargs = self._build_kwargs(
            stream=False,
            enable_thinking=enable_thinking,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        )
        request_kwargs["messages"] = converted_messages
        if system:
            request_kwargs["system"] = system  # Anthropic 的 system 是独立参数

        try:
            # 【知识点：Anthropic 的 API 入口】
            # client.messages.create() 对应 HTTP 请求：
            #   POST {base_url}/v1/messages
            # 注意：是 messages.create()，不是 chat.completions.create()
            response = self.client.messages.create(**request_kwargs)
        except Exception as e:
            logger.error(f"[{self.model_name}] Anthropic API 调用失败: {e}")
            raise

        # 【知识点：Anthropic 响应结构 — Content Blocks】
        # 与 OpenAI 不同，Anthropic 的响应是一个 content blocks 列表：
        #   response.content = [
        #       {type: "thinking", thinking: "让我想想..."},   ← 思考过程
        #       {type: "text", text: "答案是..."},           ← 最终回答
        #       {type: "tool_use", id: "...", name: "..."},  ← 工具调用
        #   ]
        # 需要遍历所有 blocks，按类型分类收集。
        content_parts = []
        reasoning_parts = []
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content_parts.append(block.text)
            elif block.type == "thinking":
                reasoning_parts.append(block.thinking)
            elif block.type == "tool_use":
                # Anthropic 的 input 已经是 dict，需转为 JSON 字符串（统一格式）
                tool_calls.append(ToolCallRequest(
                    id=block.id,
                    name=block.name,
                    arguments=json.dumps(block.input, ensure_ascii=False),
                ))

        # 【格式差异】token 统计字段名不同
        # OpenAI:    prompt_tokens / completion_tokens
        # Anthropic: input_tokens / output_tokens
        # 我们统一转换为 OpenAI 格式（方便上层代码统一处理）
        usage = {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
        }

        return ChatResult(
            content="".join(content_parts) if content_parts else None,
            reasoning="".join(reasoning_parts) if reasoning_parts else None,
            usage=usage,
            finish_reason=response.stop_reason,  # Anthropic 用 stop_reason（非 finish_reason）
            tool_calls=tool_calls if tool_calls else None,
        )

    # ==================== 流式调用 ====================

    def chat_stream(self, messages: list[dict], **kwargs) -> Generator[StreamChunk, None, None]:
        """流式调用

        【知识点：Anthropic 流式事件类型】
        Anthropic 的流式响应基于事件驱动（Event-Driven），主要事件：
            - content_block_start:  新 block 开始（text / thinking / tool_use）
            - content_block_delta:  block 增量数据（我们主要处理这个）
            - content_block_stop:   block 结束
            - message_stop:         整个消息结束（此时可获取 usage）

        【与 OpenAI 流式的差异】
        OpenAI:    每个 chunk 有 choices[0].delta.content / .reasoning_content
        Anthropic: 每个 event 有 delta.text 或 delta.thinking
        我们统一转换为 StreamChunk(type="reasoning"/"content"/"done")

        【知识点：with 语句（上下文管理器）】
        with self.client.messages.stream(...) as stream:
            ...
        with 语句确保：
            - 正常结束：关闭 HTTP 连接
            - 发生异常：也会关闭连接（不会泄露资源）
        等价于 try/finally 中手动关闭，但更简洁安全。
        """
        tools = kwargs.pop("tools", None)
        tool_choice = kwargs.pop("tool_choice", None)
        enable_thinking = kwargs.pop("enable_thinking", None)
        temperature = kwargs.pop("temperature", None)
        max_tokens = kwargs.pop("max_tokens", None)

        system, converted_messages = self._split_system_and_messages(messages)

        request_kwargs = self._build_kwargs(
            stream=True,
            enable_thinking=enable_thinking,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        )
        request_kwargs["messages"] = converted_messages
        if system:
            request_kwargs["system"] = system

        try:
            with self.client.messages.stream(**request_kwargs) as stream:
                for event in stream:
                    # 根据事件类型分发处理
                    if hasattr(event, "type"):
                        if event.type == "content_block_delta":
                            # 增量数据：可能是思考内容或回答内容
                            if hasattr(event.delta, "thinking"):
                                yield StreamChunk(type="reasoning", data=event.delta.thinking)
                            elif hasattr(event.delta, "text"):
                                yield StreamChunk(type="content", data=event.delta.text)
                        elif event.type == "message_stop":
                            # 消息结束：获取最终 usage 统计
                            final = stream.get_final_message()
                            usage = {
                                "prompt_tokens": final.usage.input_tokens,
                                "completion_tokens": final.usage.output_tokens,
                                "total_tokens": final.usage.input_tokens + final.usage.output_tokens,
                            }
                            yield StreamChunk(type="done", data=None, usage=usage)
        except Exception as e:
            logger.error(f"[{self.model_name}] Anthropic API 流式调用失败: {e}")
            raise
