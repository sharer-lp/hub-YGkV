"""
LLM 客户端子包

包含：
    - base.py             抽象基类（统一接口）
    - openai_client.py    OpenAI 格式客户端（DeepSeek / Qwen / Kimi）
    - anthropic_client.py Anthropic 格式客户端（Claude）
"""

from advanced_llm.clients.base import BaseLLMClient

__all__ = ["BaseLLMClient"]
