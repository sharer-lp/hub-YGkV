"""
LLM 客户端工厂 — 根据配置自动创建正确的客户端

==================== 设计说明 ====================

【本文件职责】
    根据模型的 provider 字段自动实例化对应的客户端：
        - provider="openai"    → OpenAIClient（DeepSeek / Qwen / Kimi）
        - provider="anthropic" → AnthropicClient（Claude）

    上层代码只需调用 create_client()，无需关心底层协议差异。

【核心设计模式：简单工厂模式（Simple Factory）】
    工厂模式的核心思想：将"对象创建"与"对象使用"分离。

    不用工厂（硬编码）：
        if provider == "openai":
            from clients.openai_client import OpenAIClient
            client = OpenAIClient("deepseek")
        elif provider == "anthropic":
            from clients.anthropic_client import AnthropicClient
            client = AnthropicClient("claude")
        # 每新增一个 provider，所有调用处都要改！

    用工厂：
        client = create_client("deepseek")  # 一行搞定，新增 provider 只改工厂
        # 调用方代码零修改！

    这就是"开闭原则"（OCP）：对扩展开放，对修改关闭。

【核心知识点：延迟导入（Lazy Import）】
    不在文件顶部 import 所有客户端，而是在需要时才 import：
        - 用户只用 DeepSeek → 只导入 openai SDK
        - 没装 anthropic SDK → 不影响 OpenAI 格式的使用
    好处：
        1. 减少启动时间（不加载不用的模块）
        2. 减少依赖（不用的 SDK 可以不装）
        3. 避免 ImportError 导致全局崩溃

    实现方式：importlib.import_module() 动态导入

用法：
    from advanced_llm import create_client

    client = create_client()            # 当前激活模型
    client = create_client("deepseek")  # → OpenAIClient
    client = create_client("qwen")      # → OpenAIClient
    client = create_client("kimi")      # → OpenAIClient
    client = create_client("claude")    # → AnthropicClient

==================================================
"""

# ==================== 导入区 ====================
# 【知识点：importlib 动态导入】
#   普通导入：from module import Class（编译时确定，必须存在）
#   动态导入：importlib.import_module("module")（运行时确定，可以失败）
#   动态导入的好处：
#     - 可以根据条件决定导入哪个模块
#     - 导入失败时可以给出友好提示（而非直接崩溃）
#     - 实现"插件化"架构：新增 provider 只需在映射表中加一行
import importlib

from advanced_llm.config import cfg, ConfigError
from advanced_llm.clients.base import BaseLLMClient


# ==================== Provider → Client 映射注册表 ====================
# 【设计思想：注册表模式（Registry Pattern）】
#   将"provider 名称"与"客户端类"的映射关系集中管理。
#   新增 provider 只需在 _PROVIDER_IMPORTS 中加一行，无需修改任何逻辑代码。
#
# 【知识点：类型注解 dict[str, type[BaseLLMClient]]】
#   type[BaseLLMClient] 表示"BaseLLMClient 的子类本身"（而非实例）。
#   例如：OpenAIClient 是 type[BaseLLMClient]，
#         OpenAIClient() 是 BaseLLMClient（实例）。
#   工厂缓存的是"类"，每次调用时再实例化。

_CLIENT_CACHE: dict[str, type[BaseLLMClient]] = {}  # 缓存已导入的客户端类（避免重复 import）

# provider 名称 → (模块路径, 类名) 的映射
# 使用元组而非直接导入，实现延迟加载
_PROVIDER_IMPORTS = {
    "openai": ("advanced_llm.clients.openai_client", "OpenAIClient"),
    "anthropic": ("advanced_llm.clients.anthropic_client", "AnthropicClient"),
    # 扩展示例（新增 provider 只需加一行）：
    # "ollama": ("advanced_llm.clients.ollama_client", "OllamaClient"),
}


def _get_client_class(provider: str) -> type[BaseLLMClient]:
    """根据 provider 获取对应客户端类（按需导入 + 缓存）

    【知识点：缓存模式（Cache Pattern）】
    第一次调用时执行 import，结果存入 _CLIENT_CACHE。
    后续调用直接从缓存取，避免重复 import 的开销。
    这是"单例 + 延迟初始化"的组合应用。

    Args:
        provider: API 格式标识（"openai" / "anthropic"）

    Returns:
        客户端类（注意是类本身，不是实例）

    Raises:
        ConfigError: provider 不支持或对应 SDK 未安装
    """
    provider = provider.lower()

    # 缓存命中：直接返回
    if provider in _CLIENT_CACHE:
        return _CLIENT_CACHE[provider]

    # 未注册的 provider
    if provider not in _PROVIDER_IMPORTS:
        raise ConfigError(
            f"❌ 不支持的 provider: '{provider}'\n"
            f"   支持的格式: {list(_PROVIDER_IMPORTS.keys())}\n"
            f"   请在 .env.models 中设置 {{MODEL}}_PROVIDER=openai 或 anthropic"
        )

    # 动态导入模块
    module_path, class_name = _PROVIDER_IMPORTS[provider]
    try:
        # 【知识点：importlib.import_module()】
        # 等价于 `import advanced_llm.clients.openai_client`，但可以在运行时动态决定导入哪个。
        # 返回模块对象，然后用 getattr 取出类。
        module = importlib.import_module(module_path)
        client_class = getattr(module, class_name)
    except ImportError as e:
        # SDK 未安装时给出友好提示（而非让用户看 traceback）
        raise ConfigError(
            f"❌ provider='{provider}' 需要安装对应 SDK\n"
            f"   错误: {e}\n"
            f"   请执行: pip install {'anthropic' if provider == 'anthropic' else 'openai'}"
        ) from e  # from e 保留原始异常链（方便调试）

    # 存入缓存
    _CLIENT_CACHE[provider] = client_class
    return client_class


# ==================== 工厂函数（对外公开 API） ====================

def create_client(model_name: str = None) -> BaseLLMClient:
    """
    创建 LLM 客户端（核心入口）

    【这是上层代码唯一需要知道的函数】
    不管底层是 OpenAI 还是 Anthropic，调用方只需要：
        client = create_client("deepseek")
        result = client.chat(messages)
    完全不需要知道 OpenAIClient / AnthropicClient 的存在。

    根据模型的 provider 配置自动选择对应格式的客户端：
        - DeepSeek / Qwen / Kimi → OpenAI 格式
        - Claude                 → Anthropic 格式

    Args:
        model_name: 模型别名（如 "deepseek", "qwen", "kimi", "claude"）
                    为 None 时使用 ACTIVE_MODEL 指定的默认模型

    Returns:
        BaseLLMClient 子类实例（统一接口，可直接调用 .chat() / .chat_stream()）
    """
    model_cfg = cfg.get_model(model_name) if model_name else cfg.active_model
    client_class = _get_client_class(model_cfg.provider)
    return client_class(model_name)  # 实例化客户端


def create_all_clients() -> dict[str, BaseLLMClient]:
    """创建所有已配置模型的客户端（用于多模型对比场景）

    【容错设计】
    某个模型配置不完整（如缺 API Key）时，跳过该模型而非整体崩溃。
    这是"优雅降级"（Graceful Degradation）的体现。

    Returns:
        {model_name: client_instance} 字典
    """
    clients = {}
    for name in cfg.available_models:
        try:
            clients[name] = create_client(name)
        except ConfigError as e:
            import logging
            logging.getLogger(__name__).warning(f"跳过模型 '{name}': {e}")
    return clients
