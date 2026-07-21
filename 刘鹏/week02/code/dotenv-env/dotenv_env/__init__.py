"""
dotenv_env — 多环境 + 多模型 + 多格式 LLM 客户端包

==================== 快速使用 ====================

    from dotenv_env import create_client, cfg

    # 创建客户端（自动识别 OpenAI / Anthropic 格式）
    client = create_client()              # 当前激活模型
    client = create_client("claude")      # 指定 Claude

    # 统一调用
    messages = [{"role": "user", "content": "你好"}]
    result = client.chat(messages)
    print(result.content)

==================== 导出清单 ====================

核心入口：
    - create_client()       创建客户端（工厂函数）
    - create_all_clients()  创建所有已配置模型的客户端
    - cfg                   全局配置单例

数据类型：
    - ChatResult            非流式响应结构
    - StreamChunk           流式 chunk 结构
    - ModelConfig           模型连接配置

客户端类（一般不直接使用，通过 create_client 创建）：
    - BaseLLMClient         抽象基类
    - OpenAIClient          OpenAI 格式
    - AnthropicClient       Anthropic 格式

异常：
    - ConfigError           配置错误

==================================================
"""

# 配置（全局单例）
from dotenv_env.config import cfg, Config, ConfigError, ModelConfig

# 数据模型
from dotenv_env.models import ChatResult, StreamChunk

# 客户端基类与实现
from dotenv_env.clients.base import BaseLLMClient
from dotenv_env.clients.openai_client import OpenAIClient
from dotenv_env.clients.anthropic_client import AnthropicClient

# 工厂函数（核心入口）
from dotenv_env.factory import create_client, create_all_clients

__all__ = [
    # 核心入口
    "create_client",
    "create_all_clients",
    "cfg",
    # 配置
    "Config",
    "ConfigError",
    "ModelConfig",
    # 数据模型
    "ChatResult",
    "StreamChunk",
    # 客户端
    "BaseLLMClient",
    "OpenAIClient",
    "AnthropicClient",
]

__version__ = "2.0.0"
