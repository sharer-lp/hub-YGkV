"""
流式输出 — 逐 chunk 展示模型回复（含 thinking 过程）
"""

from openai import OpenAI

client = OpenAI(
    api_key="sk-879e6628fec6417cbfd6b69c3e4d6ac0",
    base_url="https://api.deepseek.com",
)

stream = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "你好，帮我介绍机器学习"},
    ],
    stream=True,
    reasoning_effort="high",
    extra_body={"thinking": {"type": "enabled"}},
)

full_content = ""
reasoning_content = ""

for chunk in stream:
    choice = chunk.choices[0] if chunk.choices else None
    if choice is None:
        # 最后一个 chunk 可能只包含 usage 信息
        if hasattr(chunk, "usage") and chunk.usage:
            print(f"\n--- Token 用量 ---")
            print(f"  prompt_tokens:     {chunk.usage.prompt_tokens}")
            print(f"  completion_tokens: {chunk.usage.completion_tokens}")
            print(f"  total_tokens:      {chunk.usage.total_tokens}")
        continue

    delta = choice.delta

    # reasoning 内容（若模型返回）
    if hasattr(delta, "reasoning_content") and delta.reasoning_content:
        reasoning_content += delta.reasoning_content
        print(delta.reasoning_content, end="", flush=True)

    # 常规内容
    if delta.content:
        full_content += delta.content
        print(delta.content, end="", flush=True)

print("\n" + "=" * 50)
print("完整回复：")
print(full_content)
