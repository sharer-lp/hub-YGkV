"""
模型参数对比 — 同一 prompt，不同参数下的效果差异

重点对比：
  - temperature:     随机性（低 → 确定，高 → 多样）
  - max_tokens:      最大输出长度
  - reasoning_effort: 思考深度（low / medium / high）
  - logprobs:        各 token 的置信度
  - stop:            提前截断
"""

from openai import OpenAI

client = OpenAI(
    api_key="sk-879e6628fec6417cbfd6b69c3e4d6ac0",
    base_url="https://api.deepseek.com",
)

PROMPT = "用一句话介绍机器学习，并给出一个生活例子。"


def chat(**kwargs) -> str:
    """执行一次 chat 调用并返回回复文本。"""
    resp = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": PROMPT},
        ],
        reasoning_effort="high",
        extra_body={"thinking": {"type": "enabled"}},
        **kwargs,
    )
    return resp.choices[0].message.content or ""


# ═══════════════════════════════════════════════════════════════════════════
# 1. temperature — 随机性
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("1️⃣  temperature 对比")
print("=" * 60)

for temp in [0.0, 0.7, 1.5]:
    text = chat(temperature=temp)
    print(f"\ntemperature={temp}")
    print(f"  {text}")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# 2. max_tokens — 输出长度限制
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("2️⃣  max_tokens 对比")
print("=" * 60)

for limit in [20, 100, 500]:
    text = chat(max_tokens=limit, temperature=0.3)
    print(f"\nmax_tokens={limit}")
    print(f"  {text}")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# 3. reasoning_effort — 思考深度
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("3️⃣  reasoning_effort 对比")
print("=" * 60)

for effort in ["low", "medium", "high"]:
    resp = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "比较一下随机森林和神经网络的优缺点"},
        ],
        reasoning_effort=effort,
        extra_body={"thinking": {"type": "enabled"}},
        temperature=0.3,
    )
    text = resp.choices[0].message.content or ""
    print(f"\nreasoning_effort={effort}")
    print(f"  {text[:]}...")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# 4. logprobs — 输出 token 的对数概率
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("4️⃣  logprobs — 查看模型生成每个 token 的置信度")
print("=" * 60)

resp = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "世界上最高的山峰是？"},
    ],
    max_tokens=30,
    temperature=0.0,
    logprobs=True,
    top_logprobs=3,
)

choice = resp.choices[0]
print(f"\n回复: {choice.message.content}\n")

lp = choice.logprobs
if lp and lp.content:
    print("前 8 个 token 的 top-3 候选（logprob 越接近 0 → 越确信）：")
    for i, t in enumerate(lp.content[:8]):
        print(f"  token {i}: 选中={t.token!r:12s}  logprob={t.logprob:.2f}")
        for alt in t.top_logprobs:
            print(f"            ├─ {alt.token!r:12s}  logprob={alt.logprob:.2f}")
else:
    print("（此模型/API 未返回 logprobs 信息）")
print()


# ═══════════════════════════════════════════════════════════════════════════
# 5. stop — 停止序列，提前截断回复
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("5️⃣  stop — 遇到指定字符串时停止生成")
print("=" * 60)

PROMPT_STOP = "写一段关于春天的描述，约50字。"


def chat_stop(prompt: str, stop: list[str] | None = None) -> str:
    resp = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=200,
        stop=stop,
    )
    return resp.choices[0].message.content or ""


text_no_stop = chat_stop(PROMPT_STOP)
print(f"\n无 stop:\n  {text_no_stop}\n")

text_stop = chat_stop(PROMPT_STOP, stop=["。"])
print(f"stop=['。'] (遇句号即停):\n  {text_stop}")

text_stop2 = chat_stop(PROMPT_STOP, stop=["春天"])
print(f"\nstop=['春天'] (遇\"春天\"即停):\n  {text_stop2}")
print()

