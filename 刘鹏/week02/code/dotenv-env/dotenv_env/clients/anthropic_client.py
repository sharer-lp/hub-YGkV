"""
Anthropic 格式 LLM 客户端

==================== 适用模型 ====================

Anthropic Claude 系列（原生 Messages API）：
    - claude-sonnet-4-20250514
    - claude-3-5-haiku-20241022
    - 其他 Claude 模型

协议特点（与 OpenAI 格式的核心差异）：
    - 使用 anthropic SDK（非 openai SDK）
    - system 消息独立传递，不放在 messages 数组中
    - 思考模式通过 thinking={"type": "enabled", "budget_tokens": N} 控制
    - 流式事件类型不同（content_block_delta / thinking_delta）
    - 响应结构为 content blocks 列表

==================================================
"""

import logging
from typing import Optional, Generator

import anthropic
from anthropic import APIError, APITimeoutError, RateLimitError

from dotenv_env.config import cfg
from dotenv_env.models import ChatResult, StreamChunk
from dotenv_env.clients.base import BaseLLMClient

logger = logging.getLogger(__name__)


class AnthropicClient(BaseLLMClient):
    """
    Anthropic 格式客户端（Claude 系列）

    核心差异处理：
        1. system 消息从 messages 中剥离，通过 system 参数传递
        2. 思考模式使用 thinking.budget_tokens 控制
        3. 流式事件解析适配 Anthropic 事件协议

    用法：
        client = AnthropicClient("claude")
        result = client.chat(messages)
        for chunk in client.chat_stream(messages): ...
    """

    # 默认思考预算 token 数（思考模式下）
    DEFAULT_THINKING_BUDGET = 10000

    def __init__(self, model_name: str = None):
        super().__init__(model_name)

        model_cfg = cfg.get_model(model_name) if model_name else cfg.active_model

        self.client = anthropic.Anthropic(
            api_key=model_cfg.api_key,
            base_url=model_cfg.base_url or None,  # Anthropic 默认可不传
            timeout=cfg.TIMEOUT,
            max_retries=cfg.MAX_RETRIES,
        )

    # ==================== 消息格式转换 ====================

    @staticmethod
    def _split_system_message(messages: list[dict]) -> tuple[Optional[str], list[dict]]:
        """
        将 system 消息从 messages 中剥离

        Anthropic 要求 system 作为独立参数，不能放在 messages 数组中

        Returns:
            (system_text, filtered_messages)
        """
        system_parts = []
        filtered = []

        for msg in messages:
            if msg["role"] == "system":
                system_parts.append(msg["content"])
            else:
                filtered.append(msg)

        system_text = "\n\n".join(system_parts) if system_parts else None
        return system_text, filtered

    # ==================== 请求参数构建 ====================

    def _build_kwargs(self, messages: list[dict], stream: bool = False) -> dict:
        """
        构建 Anthropic Messages API 请求参数

        关键差异：
            - system 独立传递
            - thinking 使用 budget_tokens 而非 type
            - max_tokens 必填
        """
        system_text, filtered_messages = self._split_system_message(messages)

        kwargs = {
            "model": self.model,
            "messages": filtered_messages,
            "max_tokens": cfg.MAX_TOKENS,
            "stream": stream,
        }

        # system 消息独立传递
        if system_text:
            kwargs["system"] = system_text

        # 思考模式配置
        if self.enable_thinking:
            # Anthropic 使用 budget_tokens 控制思考深度
            # budget_tokens 必须 < max_tokens
            budget = min(self.DEFAULT_THINKING_BUDGET, cfg.MAX_TOKENS - 1024)
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": budget,
            }
            # 思考模式下不能设置 temperature
        else:
            kwargs["temperature"] = 0.7

        return kwargs

    # ==================== 响应解析 ====================

    @staticmethod
    def _parse_content_blocks(content_blocks) -> tuple[str, Optional[str]]:
        """
        解析 Anthropic 响应的 content blocks

        Anthropic 返回格式：
            content: [
                {"type": "thinking", "thinking": "..."},   # 思考块
                {"type": "text", "text": "..."},           # 回答块
            ]

        Returns:
            (text_content, thinking_content)
        """
        text_parts = []
        thinking_parts = []

        for block in content_blocks:
            if block.type == "thinking":
                thinking_parts.append(block.thinking)
            elif block.type == "text":
                text_parts.append(block.text)

        content = "".join(text_parts)
        reasoning = "".join(thinking_parts) if thinking_parts else None

        return content, reasoning

    @staticmethod
    def _parse_usage(usage) -> Optional[dict]:
        """解析 Anthropic usage 为统一格式"""
        if not usage:
            return None
        return {
            "prompt_tokens": getattr(usage, "input_tokens", 0),
            "completion_tokens": getattr(usage, "output_tokens", 0),
            "total_tokens": (
                getattr(usage, "input_tokens", 0) +
                getattr(usage, "output_tokens", 0)
            ),
        }

    # ==================== 非流式调用 ====================

    def chat(self, messages: list[dict]) -> ChatResult:
        """
        非流式调用

        Returns:
            ChatResult 统一结构
        """
        kwargs = self._build_kwargs(messages, stream=False)

        try:
            response = self.client.messages.create(**kwargs)
        except (APITimeoutError, RateLimitError, APIError) as e:
            logger.error(f"[{self.model_name}] Anthropic API 调用失败: {e}")
            raise

        content, reasoning = self._parse_content_blocks(response.content)
        usage = self._parse_usage(response.usage)

        result = ChatResult(
            content=content,
            reasoning=reasoning,
            usage=usage,
            finish_reason=response.stop_reason,
        )

        logger.info(
            f"[{self.model_name}] 非流式响应 | model={self.model} | "
            f"tokens={usage}"
        )

        return result

    # ==================== 流式调用 ====================

    def chat_stream(self, messages: list[dict]) -> Generator[StreamChunk, None, None]:
        """
        流式调用

        Anthropic 流式事件类型：
            - message_start         → 消息开始
            - content_block_start   → 内容块开始（含 block 类型）
            - content_block_delta   → 内容增量
                - thinking_delta    → 思考增量
                - text_delta        → 文本增量
            - content_block_stop    → 内容块结束
            - message_delta         → 消息级增量（含 usage）
            - message_stop          → 消息结束

        Yields:
            StreamChunk 统一结构
        """
        kwargs = self._build_kwargs(messages, stream=True)

        try:
            with self.client.messages.stream(**kwargs) as stream:
                for event in stream:
                    # 内容块增量
                    if event.type == "content_block_delta":
                        delta = event.delta

                        # 思考增量
                        if hasattr(delta, "type") and delta.type == "thinking_delta":
                            yield StreamChunk(type="reasoning", data=delta.thinking)

                        # 文本增量
                        elif hasattr(delta, "type") and delta.type == "text_delta":
                            yield StreamChunk(type="content", data=delta.text)

                    # 消息结束（含最终 usage）
                    elif event.type == "message_delta":
                        usage = self._parse_usage(
                            getattr(event, "usage", None)
                        )
                        yield StreamChunk(type="done", data=None, usage=usage)

        except (APITimeoutError, RateLimitError, APIError) as e:
            logger.error(f"[{self.model_name}] Anthropic API 流式调用失败: {e}")
            raise
