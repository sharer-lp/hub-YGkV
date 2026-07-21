"""
多环境 + 多模型 演示入口

运行方式：
    # 默认开发环境 + 默认模型
    python main.py

    # 生产环境
    $env:APP_ENV="prod"; python main.py

    # 临时切换模型
    $env:ACTIVE_MODEL="qwen"; python main.py
"""

import logging
from config import cfg
from deepseek_client import DeepSeekClient

# 日志配置
logging.basicConfig(
    level=getattr(logging, cfg.LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def demo_non_stream():
    """非流式调用示例（使用当前激活模型）"""
    client = DeepSeekClient()

    messages = [
        {"role": "system", "content": "你是一个专业的 Python 开发者"},
        {"role": "user", "content": "用一句话解释 Python 的 GIL"},
    ]

    result = client.chat(messages)

    if result["reasoning"]:
        print("【思考过程】")
        print(result["reasoning"][:200] + "..." if len(result["reasoning"]) > 200 else result["reasoning"])
        print()

    print("【回答】")
    print(result["content"])


def demo_stream():
    """流式调用示例"""
    client = DeepSeekClient()

    messages = [
        {"role": "user", "content": "9.11 和 9.8 哪个大？"},
    ]

    print("【流式输出】")
    result = client.chat_stream_print(messages)
    print(f"\nToken 消耗: {result['usage']}")


def demo_multi_model():
    """
    多模型切换示例
    演示如何在代码中显式指定不同模型
    """
    messages = [
        {"role": "user", "content": "你好，请用一句话介绍你自己"},
    ]

    # 遍历所有已配置的模型
    for model_name in cfg.available_models:
        print(f"\n{'='*50}")
        print(f"  模型: {model_name}")
        print(f"{'='*50}")

        try:
            client = DeepSeekClient(model_name=model_name)
            result = client.chat(messages)
            print(f"  回答: {result['content']}")
        except Exception as e:
            print(f"  ❌ 调用失败: {e}")


def demo_env_info():
    """打印当前环境和配置信息"""
    print(cfg)
    print()
    print(f"  当前环境: {cfg.APP_ENV}")
    print(f"  激活模型: {cfg.active_model.name} ({cfg.active_model.model})")
    print(f"  可用模型: {cfg.available_models}")
    print(f"  DEBUG: {cfg.DEBUG}")
    print(f"  LOG_LEVEL: {cfg.LOG_LEVEL}")
    print()


if __name__ == "__main__":
    demo_env_info()

    # 取消注释运行对应示例
    demo_non_stream()
    # demo_stream()
    # demo_multi_model()
