"""
LLM 客户端抽象基类

==================== 设计说明 ====================

所有格式的客户端（OpenAI / Anthropic）必须继承 BaseLLMClient，
对外暴露统一的调用接口，上层业务代码无需关心底层协议差异。

统一返回格式：
    非流式 → ChatResult
    流式   → yield StreamChunk

设计模式：模板方法模式
    - 抽象方法（子类必须实现）：chat(), chat_stream()
    - 通用方法（子类可复用）：chat_stream_print(), build_assistant_message()

==================================================
"""

from abc import ABC, abstractmethod
from typing import Optional, Generator

from dotenv_env.models import ChatResult, StreamChunk


class BaseLLMClient(ABC):
    """
    LLM 客户端统一接口

    子类必须实现：
        - chat()          非流式调用
        - chat_stream()   流式调用（生成器）

    子类可选覆盖：
        - chat_stream_print()  流式打印（默认实现已通用）
        - build_assistant_message()  多轮对话消息构建
    """

    def __init__(self, model_name: str = None):
        """
        Args:
            model_name: 模型别名（如 "deepseek", "qwen", "claude"）
                        为 None 时使用 ACTIVE_MODEL 指定的默认模型
        """
        from dotenv_env.config import cfg
        model_cfg = cfg.get_model(model_name) if model_name else cfg.active_model
        self.model = model_cfg.model
        self.model_name = model_cfg.name
        self.provider = model_cfg.provider
        self.enable_thinking = cfg.ENABLE_THINKING

    # ==================== 必须实现 ====================

    @abstractmethod
    def chat(self, messages: list[dict]) -> ChatResult:
        """
        非流式调用

        Args:
            messages: [{"role": "system"|"user"|"assistant", "content": "..."}]

        Returns:
            ChatResult 统一结构
        """
        ...

    @abstractmethod
    def chat_stream(self, messages: list[dict]) -> Generator[StreamChunk, None, None]:
        """
        流式调用

        Args:
            messages: 同上

        Yields:
            StreamChunk 统一结构
        """
        ...

    # ==================== 通用实现（可覆盖） ====================

    def chat_stream_print(self, messages: list[dict]) -> ChatResult:
        """
        流式调用并实时打印，返回完整结果
        通用实现，子类一般无需覆盖
        """
        reasoning_parts = []
        content_parts = []
        usage = None
        is_answering = False

        for chunk in self.chat_stream(messages):
            if chunk.type == "reasoning":
                print(chunk.data, end="", flush=True)
                reasoning_parts.append(chunk.data)

            elif chunk.type == "content":
                if not is_answering:
                    if reasoning_parts:
                        print("\n")  # 思考与回答之间换行
                    is_answering = True
                print(chunk.data, end="", flush=True)
                content_parts.append(chunk.data)

            elif chunk.type == "done":
                usage = chunk.usage

        print()  # 结尾换行

        return ChatResult(
            content="".join(content_parts),
            reasoning="".join(reasoning_parts) if reasoning_parts else None,
            usage=usage,
        )

    @staticmethod
    def build_assistant_message(result: ChatResult) -> dict:
        """
        构建用于多轮对话拼接的 assistant 消息
        关键：剥离 reasoning，只保留 content
        """
        return {
            "role": "assistant",
            "content": result.content,
        }

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"model={self.model!r}, "
            f"provider={self.provider!r}, "
            f"thinking={self.enable_thinking})"
        )
