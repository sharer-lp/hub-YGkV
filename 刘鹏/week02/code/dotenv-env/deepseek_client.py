import logging
from typing import Optional
from openai import OpenAI, APIError, APITimeoutError, RateLimitError

from config import cfg

logger = logging.getLogger(__name__)


class DeepSeekClient:
    """
    LLM 客户端封装（兼容 OpenAI 协议的所有模型）
    - 支持多模型切换（deepseek / qwen / openai）
    - 支持思考模式（reasoning_content 安全访问）
    - 支持流式 / 非流式
    - 自动重试 + 错误处理
    - 多轮对话管理（自动剥离 reasoning_content）
    """

    def __init__(self, model_name: str = None):
        """
        Args:
            model_name: 模型别名（如 "deepseek", "qwen", "openai"）
                        为 None 时使用 ACTIVE_MODEL 指定的默认模型
        """
        # 获取模型配置：指定模型 or 当前激活模型
        model_cfg = cfg.get_model(model_name) if model_name else cfg.active_model

        self.client = OpenAI(
            api_key=model_cfg.api_key,
            base_url=model_cfg.base_url,
            timeout=cfg.TIMEOUT,
            max_retries=cfg.MAX_RETRIES,
        )
        self.model = model_cfg.model
        self.model_name = model_cfg.name
        self.enable_thinking = cfg.ENABLE_THINKING

    def _build_extra_body(self) -> dict:
        """
        构建 extra_body，控制思考模式
        DeepSeek V4 系列（2026-07 当前全部在线模型）使用 thinking.type 参数：
          - {"type": "enabled"}  开启思考
          - {"type": "disabled"} 关闭思考
        注意：deepseek-chat / deepseek-reasoner 已于 2026/07/24 停用
        """
        extra = {}
        if self.enable_thinking:
            extra["thinking"] = {"type": "enabled"}
        else:
            # 显式关闭思考，避免 V4 默认开启
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

        # 思考模式下不支持 temperature / top_p，不传
        if not self.enable_thinking:
            kwargs["temperature"] = 0.7

        extra = self._build_extra_body()
        if extra:
            kwargs["extra_body"] = extra

        # 思考强度控制（OpenAI 标准参数，仅思考模式生效）
        # 可选值：low / medium / high / max
        if self.enable_thinking:
            kwargs["reasoning_effort"] = cfg.REASONING_EFFORT

        return kwargs

    @staticmethod
    def extract_reasoning(message) -> Optional[str]:
        """
        安全提取 reasoning_content（非流式）
        不管字段是否存在、model_extra 是否为 None，都不会报错
        """
        return getattr(message, "reasoning_content", None)

    @staticmethod
    def extract_delta_reasoning(delta) -> Optional[str]:
        """
        安全提取流式 delta 中的 reasoning_content
        """
        return getattr(delta, "reasoning_content", None)

    @staticmethod
    def build_assistant_message(response_message) -> dict:
        """
        构建用于多轮对话拼接的 assistant 消息
        关键：剥离 reasoning_content，只保留 content
        DeepSeek 官方要求：非工具调用场景下，reasoning_content 不参与上下文拼接
        """
        return {
            "role": "assistant",
            "content": response_message.content,
        }

    # ==================== 非流式调用 ====================

    def chat(self, messages: list) -> dict:
        """
        非流式调用，返回结构化结果

        Returns:
            {
                "content": "最终回答",
                "reasoning": "思考过程（可能为 None）",
                "usage": {...},
            }
        """
        kwargs = self._build_kwargs(stream=False)
        kwargs["messages"] = messages

        try:
            response = self.client.chat.completions.create(**kwargs)
        except (APITimeoutError, RateLimitError, APIError) as e:
            logger.error(f"DeepSeek API 调用失败: {e}")
            raise

        choice = response.choices[0]
        message = choice.message

        result = {
            "content": message.content,
            "reasoning": self.extract_reasoning(message),
            "usage": response.usage,
            "finish_reason": choice.finish_reason,
        }

        logger.info(
            f"非流式响应 | model={self.model} | "
            f"prompt_tokens={response.usage.prompt_tokens} | "
            f"completion_tokens={response.usage.completion_tokens}"
        )

        return result

    # ==================== 流式调用 ====================

    def chat_stream(self, messages: list):
        """
        流式调用，yield 每个 chunk 的结构化数据

        Yields:
            {
                "type": "reasoning" | "content" | "done",
                "data": "文本片段" | None,
                "usage": {...} | None,  # 仅 done 时有值
            }
        """
        kwargs = self._build_kwargs(stream=True)
        kwargs["messages"] = messages

        try:
            stream = self.client.chat.completions.create(**kwargs)
        except (APITimeoutError, RateLimitError, APIError) as e:
            logger.error(f"DeepSeek API 流式调用失败: {e}")
            raise

        for chunk in stream:
            # 最后一个 chunk 可能没有 choices，只有 usage
            if not chunk.choices:
                yield {
                    "type": "done",
                    "data": None,
                    "usage": chunk.usage,
                }
                continue

            delta = chunk.choices[0].delta

            # 思考阶段：reasoning_content 有值
            reasoning = self.extract_delta_reasoning(delta)
            if reasoning:
                yield {
                    "type": "reasoning",
                    "data": reasoning,
                    "usage": None,
                }

            # 回答阶段：content 有值
            if delta.content:
                yield {
                    "type": "content",
                    "data": delta.content,
                    "usage": None,
                }

    def chat_stream_print(self, messages: list) -> dict:
        """
        流式调用并实时打印，返回完整结果

        Returns:
            {
                "content": "完整回答",
                "reasoning": "完整思考过程（可能为 None）",
                "usage": {...},
            }
        """
        reasoning_parts = []
        content_parts = []
        usage = None
        is_answering = False

        for chunk in self.chat_stream(messages):
            if chunk["type"] == "reasoning":
                print(chunk["data"], end="", flush=True)
                reasoning_parts.append(chunk["data"])

            elif chunk["type"] == "content":
                if not is_answering:
                    print("\n")  # 思考与回答之间换行
                    is_answering = True
                print(chunk["data"], end="", flush=True)
                content_parts.append(chunk["data"])

            elif chunk["type"] == "done":
                usage = chunk["usage"]

        print()  # 结尾换行

        if usage:
            logger.info(
                f"流式响应完成 | prompt_tokens={usage.prompt_tokens} | "
                f"completion_tokens={usage.completion_tokens}"
            )

        return {
            "content": "".join(content_parts),
            "reasoning": "".join(reasoning_parts) if reasoning_parts else None,
            "usage": usage,
        }