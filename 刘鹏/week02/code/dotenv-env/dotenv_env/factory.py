"""
LLM 客户端工厂

==================== 设计说明 ====================

根据模型的 provider 字段自动实例化对应的客户端：
    - provider="openai"    → OpenAIClient（DeepSeek / Qwen / OpenAI）
    - provider="anthropic" → AnthropicClient（Claude）

上层代码只需调用 create_client()，无需关心底层协议差异。

设计模式：简单工厂 + 延迟导入
    - 简单工厂：根据 provider 字符串返回对应实例
    - 延迟导入：只导入实际使用的 SDK，未安装 anthropic 不影响 OpenAI 格式

用法：
    from dotenv_env import create_client

    # 使用当前激活模型（自动识别格式）
    client = create_client()

    # 指定模型（自动识别格式）
    client = create_client("deepseek")   # → OpenAIClient
    client = create_client("qwen")       # → OpenAIClient
    client = create_client("claude")     # → AnthropicClient

==================================================
"""

import importlib

from dotenv_env.config import cfg, ConfigError
from dotenv_env.clients.base import BaseLLMClient


# ==================== Provider → Client 映射注册表 ====================

# 缓存已加载的客户端类（避免重复 import）
_CLIENT_CACHE: dict[str, type[BaseLLMClient]] = {}

# provider → 模块路径 + 类名（用于延迟导入）
_PROVIDER_IMPORTS = {
    "openai": ("dotenv_env.clients.openai_client", "OpenAIClient"),
    "anthropic": ("dotenv_env.clients.anthropic_client", "AnthropicClient"),
}


def _get_client_class(provider: str) -> type[BaseLLMClient]:
    """
    根据 provider 获取对应客户端类（按需导入）

    支持的 provider：
        - "openai"    → OpenAIClient
        - "anthropic" → AnthropicClient

    优势：只导入实际使用的 SDK，未安装 anthropic 不影响 OpenAI 格式模型
    """
    provider = provider.lower()

    # 已缓存，直接返回
    if provider in _CLIENT_CACHE:
        return _CLIENT_CACHE[provider]

    # 未支持的 provider
    if provider not in _PROVIDER_IMPORTS:
        raise ConfigError(
            f"❌ 不支持的 provider: '{provider}'\n"
            f"   支持的格式: {list(_PROVIDER_IMPORTS.keys())}\n"
            f"   请在 .env.models 中设置 {{MODEL}}_PROVIDER=openai 或 anthropic"
        )

    # 按需导入对应模块
    module_path, class_name = _PROVIDER_IMPORTS[provider]
    try:
        module = importlib.import_module(module_path)
        client_class = getattr(module, class_name)
    except ImportError as e:
        raise ConfigError(
            f"❌ provider='{provider}' 需要安装对应 SDK\n"
            f"   错误: {e}\n"
            f"   请执行: pip install {'anthropic' if provider == 'anthropic' else 'openai'}"
        ) from e

    # 缓存
    _CLIENT_CACHE[provider] = client_class
    return client_class


# ==================== 工厂函数 ====================

def create_client(model_name: str = None) -> BaseLLMClient:
    """
    创建 LLM 客户端（核心入口）

    根据模型的 provider 配置自动选择对应格式的客户端：
        - DeepSeek / Qwen → OpenAI 格式
        - Claude          → Anthropic 格式

    Args:
        model_name: 模型别名（如 "deepseek", "qwen", "claude"）
                    为 None 时使用 ACTIVE_MODEL 指定的默认模型

    Returns:
        BaseLLMClient 子类实例（统一接口）

    Raises:
        ConfigError: 模型未配置或 provider 不支持

    用法：
        client = create_client()            # 当前激活模型
        client = create_client("qwen")      # 指定千问
        client = create_client("claude")    # 指定 Claude

        # 统一调用（无需关心底层格式）
        result = client.chat(messages)
        for chunk in client.chat_stream(messages): ...
    """
    # 获取模型配置
    model_cfg = cfg.get_model(model_name) if model_name else cfg.active_model

    # 根据 provider 获取客户端类
    client_class = _get_client_class(model_cfg.provider)

    # 实例化
    return client_class(model_name)


def create_all_clients() -> dict[str, BaseLLMClient]:
    """
    创建所有已配置模型的客户端

    Returns:
        {模型别名: 客户端实例}

    用法：
        clients = create_all_clients()
        for name, client in clients.items():
            result = client.chat(messages)
    """
    clients = {}
    for name in cfg.available_models:
        try:
            clients[name] = create_client(name)
        except ConfigError as e:
            # 跳过配置不完整的模型
            import logging
            logging.getLogger(__name__).warning(f"跳过模型 '{name}': {e}")
    return clients
