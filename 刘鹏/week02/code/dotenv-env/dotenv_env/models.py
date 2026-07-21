"""
数据模型定义

==================== 设计说明 ====================

将所有数据结构（dataclass）集中管理，与业务逻辑分离。
好处：
    1. 避免循环导入（clients 和 factory 都依赖这些类型）
    2. 新增响应字段只改这一个文件
    3. 类型定义清晰，IDE 自动补全友好

包含：
    - ChatResult    非流式调用统一返回
    - StreamChunk   流式调用单个 chunk

==================================================
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ChatResult:
    """
    非流式调用统一返回结构

    Attributes:
        content:       模型回答文本
        reasoning:     思考过程（开启 thinking 时才有）
        usage:         token 消耗统计
        finish_reason: 结束原因（stop / length / tool_calls 等）
    """
    content: str
    reasoning: Optional[str] = None
    usage: Optional[dict] = None
    finish_reason: Optional[str] = None

    def to_dict(self) -> dict:
        """转为字典（方便 JSON 序列化）"""
        return {
            "content": self.content,
            "reasoning": self.reasoning,
            "usage": self.usage,
            "finish_reason": self.finish_reason,
        }


@dataclass
class StreamChunk:
    """
    流式调用单个 chunk

    Attributes:
        type:  chunk 类型 → "reasoning" | "content" | "done"
        data:  增量文本（reasoning 或 content 阶段）
        usage: token 统计（仅 done 阶段有值）
    """
    type: str       # "reasoning" | "content" | "done"
    data: Optional[str] = None
    usage: Optional[dict] = None

    def to_dict(self) -> dict:
        return {"type": self.type, "data": self.data, "usage": self.usage}
