import os
from openai import OpenAI

client = OpenAI(
    api_key="sk-879e6628fec6417cbfd6b69c3e4d6ac0",
    base_url="https://api.deepseek.com")

# ─────────────────────────────────────────────────────────
# choices 知识说明：
#
# response.choices 是一个列表，每个元素是一个独立的回答。
# 由参数 n 控制返回几个回答，默认 n=1（只有一个回答）。
#
#   n=1  → choices[0]         唯一的回答（日常使用的情况）
#   n=3  → choices[0/1/2]     3 个独立回答，可对比选择最佳
#
# 配合 temperature 使用效果更好：
#   temperature=0.0 → 回答确定性高，多个 choices 几乎相同
#   temperature=1.0+ → 回答多样性高，多个 choices 差异明显
#
# 注意：DeepSeek API 对 n>1 的支持可能有限，
#       OpenAI 官方 API 完整支持。
# ─────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────
# messages 知识说明：
#
# messages 是一个列表，每个元素是一条对话消息，包含：
#   - role:    消息角色（决定“谁在说话”）
#   - content: 消息内容（文本字符串）
#
# 四种角色：
#   ┌───────────┬────────────────────────────────────────────────┐
#   │ role      │ 作用                                         │
#   ├───────────┼────────────────────────────────────────────────┤
#   │ system    │ 系统指令（系统提示词或者系统消息），设定模型行为/人设（通常只有一条），具备最高优先级，且在多轮对话中始终保留在消息列表的开头  │
#   │ user      │ 用户发言，提问或提供上下文                     │
#   │ assistant │ 模型之前的回答（用于多轮对话历史）       │
#   │ tool      │ 工具调用结果（配合 function calling 使用）│
#   └───────────┴────────────────────────────────────────────────┘
#  对话上下文 = 【系统上下文】 + 【历史User Message】 + 【历史 Assistant Message】 + 【当前User Message】
#  上下文窗口 = 系统上下文 + 所有历史 Message + 当前输入 + 模型将要生成的输出。这些加起来的 Token 数不能超过模型的上下文窗口

# 关键规则：
#   1. system 可选但推荐有，且通常放在第一条
#   2. messages 必须至少包含一条 user 消息
#   3. 多轮对话 = 把历史消息全部传给 messages，模型据此保持上下文
#   4. content 可以是字符串，也可以是对象列表（多模态场景）
# ─────────────────────────────────────────────────────────

# ═══════════════════════════════════════════════════════
# 示例 1：单个回答（默认 n=1）
# ═══════════════════════════════════════════════════════

response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[  # type: ignore[arg-type]
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "用一句话解释什么是机器学习"},
    ],
    stream=False,
    reasoning_effort="high",  # type: ignore[arg-type]
    extra_body={"thinking": {"type": "enabled"}}
)

print(f"返回了 {len(response.choices)} 个回答")  # 默认 1 个
print(f"choices[0] 的 finish_reason: {response.choices[0].finish_reason}")
print()
print("【最终回复】")
print(response.choices[0].message.content)

# 思考过程（reasoning_content 在 model_extra 中）
message = response.choices[0].message
reasoning = getattr(message, "reasoning_content", None) or (
    message.model_extra.get("reasoning_content") if message.model_extra else None
)

if reasoning:
    print("\n【思考过程】")
    print(reasoning[:200] + ("..." if len(reasoning) > 200 else ""))

print()

# ═══════════════════════════════════════════════════════
# 示例 2：多轮对话（assistant 角色串联上下文）
# ═══════════════════════════════════════════════════════

print("=" * 50)
print("多轮对话演示")
print("=" * 50)

# 模拟一个多轮对话历史
conversation = [  # type: ignore[arg-type]
    {"role": "system", "content": "你是一个Python编程助手，回答要简洁。"},
    {"role": "user", "content": "Python中列表和元组有什么区别？"},
    {"role": "assistant", "content": "列表可变（可增删改），元组不可变（创建后不能修改）。列表用[]，元组用()。"},
    {"role": "user", "content": "那字典呢？"},
]

resp_multi_turn = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=conversation,
    stream=False,
    max_tokens=200,
)

assistant_reply = resp_multi_turn.choices[0].message.content
print(f"\n用户问: {conversation[-1]['content']}")
print(f"模型答: {assistant_reply}")

# 把模型回答追加到历史，供下一轮使用
# 大模型API是无状态的，每次请求都是失忆状态，必须手动将之前的所有的对话历史（system提示词、历史user提问、历史assistant回答）以及本次的user提问，都放在messages参数中，才能保持上下文。

# 解决上下文过大的问题

# 1.滑动窗口截断：只保留最近N轮对话，把更早的消息直接删除。但这会导致模型遗忘很久之前说过的话。
# 2.摘要总结：当历史超过一定长度时，让模型先对之前的对话总结一段摘要，然后把这段摘要作为新的system提示，丢给原始的明细对话。
# 3.向量数据库（RAG): 使用向量数据库存储历史对话，通过向量检索找到最相关的对话片段，然后把这些片段作为context输入给模型。
conversation.append({"role": "assistant", "content": assistant_reply})
print(f"\n当前对话历史共 {len(conversation)} 条消息")
print("（实际项目中，历史过长会消耗更多 token，需做截断或摘要）")

print()

# ═══════════════════════════════════════════════════════
# 示例 3：多个回答对比（n=3，需要较高 temperature）
# ═══════════════════════════════════════════════════════
# 注：DeepSeek 可能不支持 n>1，如报错可注释掉此段

print("=" * 50)
print("尝试获取多个回答 (n=3)...")
try:
    response_multi = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[  # type: ignore[arg-type]
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "用一句话解释什么是机器学习"},
        ],
        n=3,              # 返回 3 个独立回答
        temperature=1.0,  # 提高多样性
        stream=False,
    )

    print(f"返回了 {len(response_multi.choices)} 个回答\n")
    for i, choice in enumerate(response_multi.choices):
        print(f"  回答 {i+1}: {choice.message.content}")
        print(f"           finish_reason: {choice.finish_reason}")
        print()

except Exception as e:
    print(f"  ⚠️  多回答请求失败: {e}")
    print("  （DeepSeek API 可能不支持 n>1，这是正常的）")


# ═══════════════════════════════════════════════════════
# 示例 4：提示词模板（Prompt Template）—— 占位符替换
# ═══════════════════════════════════════════════════════
# 核心思路：
#   将 system prompt 设计成模板，关键信息用占位符，
#   用户提问时动态替换，实现“同一个角色，不同输入”的复用。
#
# 常用占位符方式：
#   1. f-string:      f"你好{name}"           （简单直接）
#   2. str.format():  "你好{name}".format(...)  （可复用模板）
#   3. str.Template:  Template("你好$name")     （更安全）
# ─────────────────────────────────────────────────────────

print()
print("=" * 50)
print("提示词模板演示")
print("=" * 50)

# -------- 模板定义（可以放在配置文件或单独模块中） --------

# 模板 1：翻译助手（用 .format() 方式，模板可复用）
TRANSLATOR_SYSTEM = """你是一位专业的翻译专家。
源语言：{source_lang}
目标语言：{target_lang}
要求：
  - 保持原文语气和风格
  - 专业术语要准确
  - 只输出翻译结果，不要解释"""

# 模板 2：代码审查（用 {xxx} 占位符）
CODE_REVIEWER_SYSTEM = """你是一位资深{language}工程师，请审查以下代码。
审查重点：
  - 代码风格和命名规范
  - 潜在的 bug 和边界情况
  - 性能优化建议
请用中文回答，语气友善。"""

# 模板 3：角色扮演（用 f-string 方式，简单场景）
def make_role_prompt(role_name: str, expertise: str, tone: str) -> str:
    """动态生成角色人设 prompt"""
    return f"""你是一位{role_name}，擅长{expertise}。
说话风格：{tone}
请始终保持这个角色来回答用户的问题。"""


# -------- 实际使用 --------

# 用法 1：翻译助手 — 替换语言占位符
print("\n—— 翻译助手 ——")
source_text = "机器学习是人工智能的一个分支，它使计算机能够从数据中学习。"

resp_translate = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[  # type: ignore[arg-type]
        {"role": "system", "content": TRANSLATOR_SYSTEM.format(
            source_lang="中文",
            target_lang="英文",
        )},
        {"role": "user", "content": source_text},
    ],
    stream=False,
    temperature=0.3,
)
print(f"原文: {source_text}")
print(f"译文: {resp_translate.choices[0].message.content}")


# 用法 2：代码审查 — 替换编程语言
print("\n—— 代码审查 ——")
code_snippet = """
def calc_avg(numbers):
    total = 0
    for n in numbers:
        total += n
    return total / len(numbers)
"""

resp_review = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[  # type: ignore[arg-type]
        {"role": "system", "content": CODE_REVIEWER_SYSTEM.format(language="Python")},
        {"role": "user", "content": f"请审查这段代码：\n```python{code_snippet}```"},
    ],
    stream=False,
    max_tokens=500,
)
print(f"审查结果:\n{resp_review.choices[0].message.content}")


# 用法 3：动态角色 — 用函数生成 prompt
print("\n—— 动态角色切换 ——")
roles = [
    ("小学老师", "用简单易懂的方式讲解概念", "亲切耐心，多用比喻"),
    ("毒舌影评人", "影视分析和吐槽", "犬儒幽默，一针见血"),
]

for role_name, expertise, tone in roles:
    system_prompt = make_role_prompt(role_name, expertise, tone)
    resp_role = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[  # type: ignore[arg-type]
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "你觉得《三体》这部小说怎么样？"},
        ],
        stream=False,
        max_tokens=300,
    )
    content = resp_role.choices[0].message.content
    finish_reason = resp_role.choices[0].finish_reason

    if content:
        print(f"\n  【{role_name}】: {content}")
    else:
        print(f"\n  【{role_name}】: ⚠️  返回内容为空")
        print(f"    finish_reason: {finish_reason}")
        print(f"    system_prompt: {system_prompt[:80]}...")