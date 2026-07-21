"""
OpenAI 格式 LLM 客户端

==================== 适用模型 ====================

所有兼容 OpenAI Chat Completions API 的模型：
    - DeepSeek（deepseek-v4-flash / deepseek-v4-pro）
    - 通义千问（qwen-plus / qwen-max / qwen-turbo）
    - OpenAI 原生（gpt-4o / o3 等）
    - 其他兼容接口（Moonshot、智谱、零一万物等）

协议特点：
    - 使用 openai SDK
    - 支持 extra_body 传递厂商私有参数（如 DeepSeek thinking）
    - reasoning_content 通过 getattr 安全访问

==================================================
"""

import logging
from typing import Optional, Generator

from openai import OpenAI, APIError, APITimeoutError, RateLimitError

from dotenv_env.config import cfg
from dotenv_env.models import ChatResult, StreamChunk
from dotenv_env.clients.base import BaseLLMClient

logger = logging.getLogger(__name__)


class OpenAIClient(BaseLLMClient):
    """
    OpenAI 格式客户端

    支持所有 OpenAI Chat Completions 兼容接口：
        - DeepSeek V4 系列（thinking.type 参数控制思考）
        - 通义千问（DashScope 兼容模式）
        - OpenAI 原生

    用法：
        client = OpenAIClient()                # 使用 ACTIVE_MODEL
        client = OpenAIClient("deepseek")      # 指定 DeepSeek
        client = OpenAIClient("qwen")          # 指定千问

        result = client.chat(messages)
        for chunk in client.chat_stream(messages): ...
    """

    def __init__(self, model_name: str = None):
        super().__init__(model_name)

        model_cfg = cfg.get_model(model_name) if model_name else cfg.active_model

        self.client = OpenAI(
            api_key=model_cfg.api_key,
            base_url=model_cfg.base_url,
            timeout=cfg.TIMEOUT,
            max_retries=cfg.MAX_RETRIES,
        )

    # ==================== 请求参数构建 ====================

    def _build_extra_body(self) -> dict:
        """
        构建 extra_body（厂商私有参数）

        DeepSeek V4 系列使用 thinking.type 控制思考模式：
            - {"type": "enabled"}  开启
            - {"type": "disabled"} 关闭（显式关闭避免 V4 默认开启）

        千问目前不需要 extra_body，预留扩展
        """
        extra = {}

        # DeepSeek 专属：thinking 控制
        if self.model_name == "deepseek":
            if self.enable_thinking:
                extra["thinking"] = {"type": "enabled"}
            else:
                extra["thinking"] = {"type": "disabled"}

        return extra

    def _build_kwargs(self, stream: bool = False) -> dict:
        """构建通用请求参数"""
        kwargs = {
            "model": self.model,
            "max_tokens": cfg.MAX_TOKENS,
            "stream": stream,
        }

        if stream:
            kwargs["stream_options"] = {"include_usage": True}

        # 思考模式下不支持 temperature / top_p
        if not self.enable_thinking:
            kwargs["temperature"] = 0.7

        # 厂商私有参数
        extra = self._build_extra_body()
        if extra:
            kwargs["extra_body"] = extra

        # 思考强度（OpenAI 标准参数，仅思考模式生效）
        if self.enable_thinking:
            kwargs["reasoning_effort"] = cfg.REASONING_EFFORT

        return kwargs

    # ==================== 安全提取方法 ====================

    @staticmethod
    def _extract_reasoning(message) -> Optional[str]:
        """安全提取 reasoning_content（非流式）"""
        return getattr(message, "reasoning_content", None)

    @staticmethod
    def _extract_delta_reasoning(delta) -> Optional[str]:
        """安全提取流式 delta 中的 reasoning_content"""
        return getattr(delta, "reasoning_content", None)

    # ==================== 非流式调用 ====================

    def chat(self, messages: list[dict]) -> ChatResult:
        """
        非流式调用

        Returns:
            ChatResult 统一结构
        """
        kwargs = self._build_kwargs(stream=False)
        kwargs["messages"] = messages

        try:
            response = self.client.chat.completions.create(**kwargs)
        except (APITimeoutError, RateLimitError, APIError) as e:
            logger.error(f"[{self.model_name}] OpenAI API 调用失败: {e}")
            raise

        choice = response.choices[0]
        message = choice.message
        usage = response.usage

        result = ChatResult(
            content=message.content,
            reasoning=self._extract_reasoning(message),
            usage={
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            } if usage else None,
            finish_reason=choice.finish_reason,
        )

        logger.info(
            f"[{self.model_name}] 非流式响应 | model={self.model} | "
            f"tokens={result.usage}"
        )

        return result

    # ==================== 流式调用 ====================

    def chat_stream(self, messages: list[dict]) -> Generator[StreamChunk, None, None]:
        """
        流式调用

        Yields:
            StreamChunk 统一结构
        """
        kwargs = self._build_kwargs(stream=True)
        kwargs["messages"] = messages

        try:
            stream = self.client.chat.completions.create(**kwargs)
        except (APITimeoutError, RateLimitError, APIError) as e:
            logger.error(f"[{self.model_name}] OpenAI API 流式调用失败: {e}")
            raise

        for chunk in stream:
            # 最后一个 chunk 可能没有 choices，只有 usage
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

            # 思考阶段
            reasoning = self._extract_delta_reasoning(delta)
            if reasoning:
                yield StreamChunk(type="reasoning", data=reasoning)

            # 回答阶段
            if delta.content:
                yield StreamChunk(type="content", data=delta.content)


# ==================== 向后兼容 ====================
# 旧代码 from dotenv_env.clients.openai_client import DeepSeekClient 仍可用
DeepSeekClient = OpenAIClient
