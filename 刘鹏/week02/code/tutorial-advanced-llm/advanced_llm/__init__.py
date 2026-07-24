"""
advanced_llm — 生产级多环境多模型多格式 LLM 客户端工具包

==================== 核心能力 ====================

1. 多环境：开发/生产环境配置隔离，APP_ENV 一键切换
2. 多格式：OpenAI / Anthropic 两种 API 格式统一抽象
3. 多模型：DeepSeek / Qwen / Kimi / Claude 动态切换
4. 多轮对话：完整上下文管理（含思考内容回传策略）
5. 工具调用：生产级 Function Calling 框架
6. 上下文管理：滑动窗口 / Token预算 / 摘要压缩 / 混合策略

==================== 快速开始 ====================

    from advanced_llm import create_client, cfg

    # 创建客户端（自动识别格式）
    client = create_client("deepseek")

    # 非流式调用
    result = client.chat([{"role": "user", "content": "你好"}])
    print(result.content)

    # 流式调用
    for chunk in client.chat_stream([{"role": "user", "content": "你好"}]):
        if chunk.type == "content":
            print(chunk.data, end="")

==================================================
"""

from advanced_llm.config import cfg, ConfigError
from advanced_llm.factory import create_client, create_all_clients
from advanced_llm.models import ChatResult, StreamChunk, ToolCallRequest, ToolCallResult
from advanced_llm.context_manager import ContextManager, HybridStrategy
from advanced_llm.tool_registry import ToolRegistry, create_default_registry

__version__ = "1.0.0"

__all__ = [
    # 配置
    "cfg",
    "ConfigError",
    # 工厂
    "create_client",
    "create_all_clients",
    # 数据模型
    "ChatResult",
    "StreamChunk",
    "ToolCallRequest",
    "ToolCallResult",
    # 上下文管理
    "ContextManager",
    "HybridStrategy",
    # 工具
    "ToolRegistry",
    "create_default_registry",
]
