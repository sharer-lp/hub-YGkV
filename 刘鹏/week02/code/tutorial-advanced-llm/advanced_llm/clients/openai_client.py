"""
OpenAI 格式 LLM 客户端 — 支持所有 OpenAI 兼容接口的模型

==================== 适用模型 ====================

所有兼容 OpenAI Chat Completions API 的模型：
    - DeepSeek（deepseek-chat / deepseek-reasoner）
    - 通义千问（qwen-plus / qwen-max / qwen-turbo）
    - Kimi / Moonshot（moonshot-v1-8k / moonshot-v1-32k）
    - OpenAI 原生（gpt-4o / o3 等）

【知识点：OpenAI 兼容模式是什么？】
    OpenAI 定义了 Chat Completions API 的事实标准：
        POST /v1/chat/completions
        请求体：{"model": "...", "messages": [...], "stream": bool, ...}
        响应体：{"choices": [{"message": {"role": "assistant", "content": "..."}}]}

    其他厂商（DeepSeek/Qwen/Kimi）实现了相同的接口格式，
    只需更换 base_url 和 api_key，就能用同一个 openai SDK 调用。
    这就是为什么一个 OpenAIClient 能服务多个厂商。

【协议特点】
    - 使用 openai SDK（pip install openai）
    - 支持 extra_body 传递厂商私有参数（如 DeepSeek thinking）
    - reasoning_content 通过 getattr 安全访问（因为不是所有模型都有）
    - 支持 tools / tool_choice 工具调用

【知识点：getattr(obj, "attr", default) 安全访问】
    普通访问：message.reasoning_content
        → 如果属性不存在，抛 AttributeError（程序崩溃）
    安全访问：getattr(message, "reasoning_content", None)
        → 如果属性不存在，返回 None（程序继续）
    DeepSeek 有 reasoning_content，但 Qwen/Kimi 没有，
    用 getattr 可以安全地"尝试获取，没有就算了"。

==================================================
"""

# ==================== 导入区 ====================
import json
import logging
from typing import Optional, Generator

# 【知识点：openai SDK 的异常体系】
#   openai SDK 定义了多种异常类型，方便精确捕获：
#   - APIError:       通用 API 错误（基类）
#   - APITimeoutError: 请求超时（网络问题或服务端慢）
#   - RateLimitError:  触发速率限制（QPS 超限，需要重试）
#   - AuthenticationError: API Key 无效
#   生产环境应该区分处理：超时可重试，认证错误不可重试。
from openai import OpenAI, APIError, APITimeoutError, RateLimitError

from advanced_llm.config import cfg
from advanced_llm.models import ChatResult, StreamChunk, ToolCallRequest
from advanced_llm.clients.base import BaseLLMClient

# 【知识点：logging 模块】
#   生产环境不用 print()，而用 logging：
#   - 可以控制级别（DEBUG/INFO/WARNING/ERROR）
#   - 可以输出到文件、控制台、远程服务
#   - 可以格式化（时间戳、模块名、行号）
#   __name__ 会自动取当前模块名（如 "advanced_llm.clients.openai_client"）
logger = logging.getLogger(__name__)


class OpenAIClient(BaseLLMClient):
    """
    OpenAI 格式客户端（支持所有 OpenAI Chat Completions 兼容接口）

    【继承关系】
        BaseLLMClient (ABC)
        └── OpenAIClient  ← 你在这里

    支持的模型：
        - DeepSeek 系列（thinking.type 参数控制思考）
        - 通义千问（DashScope 兼容模式）
        - Kimi / Moonshot
        - OpenAI 原生

    用法：
        client = OpenAIClient()                # 使用 ACTIVE_MODEL
        client = OpenAIClient("deepseek")      # 指定 DeepSeek
        client = OpenAIClient("qwen")          # 指定千问
        client = OpenAIClient("kimi")          # 指定 Kimi

        result = client.chat(messages)                    # 非流式
        result = client.chat(messages, tools=TOOLS)       # 带工具调用
        for chunk in client.chat_stream(messages): ...    # 流式
    """

    def __init__(self, model_name: str = None):
        """
        初始化 OpenAI 客户端

        【知识点：super().__init__(model_name)】
        调用父类 BaseLLMClient 的 __init__，完成通用初始化（读取配置、设置模型参数）。
        然后子类补充自己的初始化（创建 OpenAI SDK 客户端实例）。
        这是继承的标准用法：先父后子。

        【知识点：OpenAI SDK 客户端初始化参数】
        - api_key:    身份认证（每次请求都会携带）
        - base_url:   API 地址（不同厂商不同，这就是"兼容模式"的关键）
        - timeout:    请求超时（秒），超过则抛 APITimeoutError
        - max_retries: 自动重试次数（网络抨动时自动重试）
        """
        super().__init__(model_name)

        model_cfg = cfg.get_model(model_name) if model_name else cfg.active_model

        # 创建 OpenAI SDK 客户端实例
        # 注意：不同厂商只是 base_url 不同，SDK 用法完全一样
        self.client = OpenAI(
            api_key=model_cfg.api_key,
            base_url=model_cfg.base_url,
            timeout=cfg.TIMEOUT,
            max_retries=cfg.MAX_RETRIES,
        )

    # ==================== 请求参数构建 ====================
    # 【设计思想：将参数构建抽离为独立方法】
    #   chat() 和 chat_stream() 都需要构建请求参数，
    #   抽离为 _build_kwargs() 避免代码重复（DRY 原则：Don't Repeat Yourself）。
    #   下划线前缀 _ 表示"私有方法"（Python 约定，非强制）。

    def _build_extra_body(self, enable_thinking: bool = None) -> dict:
        """
        构建 extra_body（厂商私有参数）

        【知识点：extra_body 是什么？】
        openai SDK 的 extra_body 参数允许传递"标准 API 之外的额外字段"。
        厂商私有功能（如 DeepSeek 的 thinking 控制）不在 OpenAI 标准中，
        但厂商服务器能识别，所以通过 extra_body "夹带"过去。

        DeepSeek 思考模式控制：
            - {"thinking": {"type": "enabled"}}   开启思考
            - {"thinking": {"type": "disabled"}}  关闭思考（显式关闭避免默认开启）

        千问/Kimi 目前不需要 extra_body，预留扩展。
        """
        extra = {}
        thinking = enable_thinking if enable_thinking is not None else self.enable_thinking

        # DeepSeek 专属：thinking 控制
        if self.model_name == "deepseek" and self.supports_thinking:
            if thinking:
                extra["thinking"] = {"type": "enabled"}
            else:
                extra["thinking"] = {"type": "disabled"}

        return extra

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
        """
        构建通用请求参数（chat 和 chat_stream 共用）

        【知识点：参数优先级设计】
        每个参数都有三层优先级：
            调用时显式传入 > 全局配置（cfg） > 硬编码默认值
        例如 temperature：
            - 调用时传 temperature=0.5 → 用 0.5
            - 没传 → 用 cfg.TEMPERATURE（.env 中配置）
            - cfg 也没有 → 用 0.7（代码默认）

        Args:
            stream: 是否流式（True 时返回生成器，False 时返回完整响应）
            enable_thinking: 是否开启思考（None 使用全局配置）
            temperature: 温度（控制随机性，0=确定性，1=创造性）
            max_tokens: 最大输出 token 数
            tools: 工具定义列表（JSON Schema 格式）
            tool_choice: 工具选择策略（"auto"/"none"/"required"/具体工具名）
            **extra_kwargs: 其他透传参数（预留扩展）
        """
        thinking = enable_thinking if enable_thinking is not None else self.enable_thinking

        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens or cfg.MAX_TOKENS,
            "stream": stream,
        }

        # 流式时请求 usage 统计（否则流式响应不包含 token 计数）
        if stream:
            kwargs["stream_options"] = {"include_usage": True}

        # ==================== temperature 参数处理 ====================
        # 【核心兼容性逻辑】
        # 不同模型对 temperature 的要求完全不同：
        #   - Kimi K2.7-code: 只允许 temperature=1.0（固定值，传其他报错）
        #   - Kimi K2.6/K2.5: 思考模式=1.0，非思考=0.6（固定值）
        #   - DeepSeek 思考模式: 不允许传 temperature
        #   - 其他模型: 正常支持自定义 temperature
        #
        # 处理策略（优先级从高到低）：
        #   1. fixed_params 中有 temperature → 使用固定值（模型硬性要求）
        #   2. 思考模式且不支持固定 temperature → 不传（DeepSeek 限制）
        #   3. 其他情况 → 使用用户传入值或全局配置
        if "temperature" in self.fixed_params:
            # 模型有固定 temperature 要求（如 Kimi K2 系列）
            kwargs["temperature"] = self.fixed_params["temperature"]
        elif not thinking:
            # 非思考模式：正常设置 temperature
            kwargs["temperature"] = temperature if temperature is not None else cfg.TEMPERATURE
        # else: 思考模式且无固定约束 → 不传 temperature（DeepSeek 限制）

        # 厂商私有参数（通过 extra_body 传递）
        extra = self._build_extra_body(enable_thinking=thinking)
        if extra:
            kwargs["extra_body"] = extra

        # 思考强度（仅对支持 reasoning_effort 参数的模型生效）
        # 【兼容性说明】
        #   - OpenAI o系列: 支持 reasoning_effort="low/medium/high"
        #   - Kimi K3: 支持 reasoning_effort="max"（仅此一个值）
        #   - Kimi K2.7-code: 不支持 reasoning_effort（传入会报错）
        #   - DeepSeek: 通过 extra_body 控制思考，不用 reasoning_effort
        # 因此只有 thinking_param_type == "reasoning_effort" 的模型才发送此参数
        if thinking and self.thinking_param_type == "reasoning_effort":
            kwargs["reasoning_effort"] = cfg.REASONING_EFFORT

        # 工具调用参数
        if tools:
            kwargs["tools"] = tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice

        # 透传其他参数（预留扩展，如 response_format、seed 等）
        kwargs.update(extra_kwargs)

        return kwargs

    # ==================== 安全提取方法 ====================
    # 【设计思想：将"从响应中提取字段"抽离为独立方法】
    #   不同模型的响应结构略有差异（有的有 reasoning_content，有的没有），
    #   抽离为独立方法可以：
    #     1. 统一处理"字段可能不存在"的情况
    #     2. 方便后续扩展（如新增字段只需改一处）
    #     3. 代码可读性更好（chat() 方法不会被提取逻辑污染）

    @staticmethod
    def _extract_reasoning(message) -> Optional[str]:
        """安全提取 reasoning_content（非流式）

        【知识点：getattr 安全访问】
        getattr(obj, "attr", default) 三参数版本：
            - 属性存在 → 返回属性值
            - 属性不存在 → 返回 default（而非抛异常）
        DeepSeek 有 reasoning_content，Qwen/Kimi 没有，
        用 getattr 可以安全地"尝试获取，没有就返回 None"。
        """
        return getattr(message, "reasoning_content", None)

    @staticmethod
    def _extract_delta_reasoning(delta) -> Optional[str]:
        """安全提取流式 delta 中的 reasoning_content

        【知识点：流式中的 delta vs 非流式的 message】
        非流式：response.choices[0].message  → 完整消息
        流式：  chunk.choices[0].delta       → 增量片段（只有新增的部分）
        delta 是拉丁语"变化量"的意思，每次只包含新生成的几个字。
        """
        return getattr(delta, "reasoning_content", None)

    @staticmethod
    def _extract_tool_calls(message) -> Optional[list[ToolCallRequest]]:
        """提取工具调用请求

        【知识点：工具调用的响应结构】
        当模型决定调用工具时，message.tool_calls 是一个列表：
            [
                {id: "call_xxx", function: {name: "get_weather", arguments: '{"city":"北京"}'}},
                {id: "call_yyy", function: {name: "calculate", arguments: '{"expr":"2+2"}'}},
            ]
        注意：arguments 是 JSON 字符串，不是字典！
        可能同时调用多个工具（并行工具调用）。
        """
        if not message.tool_calls:
            return None
        return [
            ToolCallRequest(
                id=tc.id,
                name=tc.function.name,
                arguments=tc.function.arguments,  # JSON 字符串，后续用 json.loads() 解析
            )
            for tc in message.tool_calls
        ]

    # ==================== 非流式调用 ====================

    def chat(self, messages: list[dict], **kwargs) -> ChatResult:
        """
        非流式调用（发送请求 → 等待完整响应 → 一次性返回）

        【调用流程】
            1. _build_kwargs() 构建请求参数
            2. self.client.chat.completions.create() 发送 HTTP 请求
            3. 解析响应，转换为统一的 ChatResult

        Args:
            messages: 对话历史
            **kwargs: 额外参数（tools, tool_choice, enable_thinking, temperature 等）

        Returns:
            ChatResult 统一结构

        Raises:
            APITimeoutError: 请求超时
            RateLimitError: 触发速率限制
            APIError: 其他 API 错误
        """
        request_kwargs = self._build_kwargs(stream=False, **kwargs)
        request_kwargs["messages"] = messages

        try:
            # 【知识点：openai SDK 的调用方式】
            # client.chat.completions.create() 对应 HTTP 请求：
            #   POST {base_url}/chat/completions
            #   Body: {"model": "...", "messages": [...], ...}
            # SDK 封装了 HTTP 请求、响应解析、错误处理、自动重试等细节。
            response = self.client.chat.completions.create(**request_kwargs)
        except (APITimeoutError, RateLimitError, APIError) as e:
            logger.error(f"[{self.model_name}] OpenAI API 调用失败: {e}")
            raise  # 记录日志后继续抛出，让上层决定如何处理

        # 解析响应结构
        # 【知识点：OpenAI 响应结构】
        # response.choices[0].message.content       → 回答文本
        # response.choices[0].finish_reason         → 结束原因
        # response.usage.prompt_tokens              → 输入 token 数
        # response.usage.completion_tokens          → 输出 token 数
        choice = response.choices[0]
        message = choice.message
        usage = response.usage

        # 转换为统一的 ChatResult（屏蔽厂商差异）
        result = ChatResult(
            content=message.content,
            reasoning=self._extract_reasoning(message),
            usage={
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            } if usage else None,
            finish_reason=choice.finish_reason,
            tool_calls=self._extract_tool_calls(message),
            raw_message=message,  # 保留原始对象，工具调用场景需要完整信息
        )

        logger.info(
            f"[{self.model_name}] 非流式响应 | model={self.model} | "
            f"finish={result.finish_reason} | tokens={result.usage}"
        )

        return result

    # ==================== 流式调用 ====================

    def chat_stream(self, messages: list[dict], **kwargs) -> Generator[StreamChunk, None, None]:
        """
        流式调用（发送请求 → 逐块接收 → 逐个 yield 出去）

        【知识点：SSE（Server-Sent Events）协议】
        流式输出基于 HTTP 的 SSE 协议：
            1. 客户端发送普通 HTTP 请求（stream=True）
            2. 服务器不关闭连接，持续发送数据块：
               data: {"choices":[{"delta":{"content":"你"}}]}
               data: {"choices":[{"delta":{"content":"好"}}]}
               data: [DONE]
            3. 客户端逐个解析数据块
        openai SDK 封装了这个过程，返回一个可迭代的生成器。

        Yields:
            StreamChunk 统一结构
        """
        request_kwargs = self._build_kwargs(stream=True, **kwargs)
        request_kwargs["messages"] = messages

        try:
            stream = self.client.chat.completions.create(**request_kwargs)
        except (APITimeoutError, RateLimitError, APIError) as e:
            logger.error(f"[{self.model_name}] OpenAI API 流式调用失败: {e}")
            raise

        # 遍历流式数据块
        for chunk in stream:
            # 【知识点：流式响应的最后一个 chunk】
            # 最后一个 chunk 可能没有 choices，只有 usage（token 统计）。
            # 这是因为设置了 stream_options={"include_usage": True}。
            if not chunk.choices:
                usage = None
                if chunk.usage:
                    usage = {
                        "prompt_tokens": chunk.usage.prompt_tokens,
                        "completion_tokens": chunk.usage.completion_tokens,
                        "total_tokens": chunk.usage.total_tokens,
                    }
                yield StreamChunk(type="done", data=None, usage=usage)
                continue

            delta = chunk.choices[0].delta

            # 思考阶段（reasoning_content 增量）
            reasoning = self._extract_delta_reasoning(delta)
            if reasoning:
                yield StreamChunk(type="reasoning", data=reasoning)

            # 回答阶段（content 增量）
            if delta.content:
                yield StreamChunk(type="content", data=delta.content)
