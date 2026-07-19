"""
JSON Output — 让模型输出结构化 JSON

适用场景：
  - 从非结构化文本中提取结构化信息
  - 后续程序需要直接解析模型输出（无需 LLM 再处理）
  - 构建自动化 pipeline

注意事项：
  1. 设置 response_format = {"type": "json_object"}
  2. prompt 中必须包含 "json" 字样和输出格式样例
  3. 合理设置 max_tokens，防止 JSON 被截断
  4. 有概率返回空 content，需做容错处理

API 参考：https://api-docs.deepseek.com/guides/json_mode
"""

import json
from openai import OpenAI

client = OpenAI(
    api_key="sk-e16dfcaa8c7e43908ae4d922f96c4a8f",
    base_url="https://api.deepseek.com",
)


# ═════════════════════════════════════════════════════════════════════════════
# 辅助函数
# ═════════════════════════════════════════════════════════════════════════════

def safe_json_parse(text: str | None) -> dict | list | None:
    """安全解析 JSON，处理可能的空 content 和格式异常。"""
    if not text or not text.strip():
        print("    ⚠️  模型返回了空 content（JSON 模式偶发问题）")
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"    ⚠️  JSON 解析失败: {e}")
        # 尝试修复常见问题：删除 markdown 代码块标记
        cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            print(f"    原始内容: {text[:200]}")
            return None


# ═════════════════════════════════════════════════════════════════════════════
# 1. 基础 JSON 输出 — 提取问答对
# ═════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("1️⃣  基础 JSON 输出 — 提取问答对")
print("=" * 65)

system_prompt = """
用户会提供一些考试题目文本。请从中解析出 "question" 和 "answer" 并以 JSON 格式输出。

输入示例：
Which is the highest mountain in the world? Mount Everest.

JSON 输出示例：
{
    "question": "Which is the highest mountain in the world?",
    "answer": "Mount Everest"
}
"""

user_prompt = "Which is the longest river in the world? The Nile River."

response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[  # type: ignore[arg-type]
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ],
    response_format={"type": "json_object"},  # type: ignore[arg-type]
    max_tokens=200,
    temperature=0.0,
)

content = response.choices[0].message.content
result = safe_json_parse(content)

if result:
    print(f"\n解析结果:")
    print(f"  question: {result['question']}")
    print(f"  answer:   {result['answer']}")
    print(f"\n原始输出:\n{json.dumps(result, ensure_ascii=False, indent=2)}")
else:
    print(f"  原始内容: {content}")

print()


# ═════════════════════════════════════════════════════════════════════════════
# 2. 复杂 JSON 输出 — 嵌套结构
# ═════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("2️⃣  复杂 JSON 输出 — 嵌套结构")
print("=" * 65)

system_prompt = """
从用户的文字描述中提取人物信息，以 JSON 格式输出，包含以下字段：
- name: 姓名
- age: 年龄（数字）
- skills: 技能列表（数组）
- address: 地址对象 {city, street}

JSON 输出示例：
{
    "name": "张三",
    "age": 28,
    "skills": ["Python", "数据分析"],
    "address": {
        "city": "北京",
        "street": "海淀区中关村大街"
    }
}
"""

user_prompt = """
李四今年32岁，是一名全栈工程师，精通JavaScript、TypeScript和React。
他目前住在上海市浦东新区张江高科技园区。
"""

response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[  # type: ignore[arg-type]
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ],
    response_format={"type": "json_object"},  # type: ignore[arg-type]
    max_tokens=500,
    temperature=0.0,
)

content = response.choices[0].message.content
result = safe_json_parse(content)

if result:
    print(f"\n解析结果:")
    print(f"  name:   {result['name']}")
    print(f"  age:    {result['age']}")
    print(f"  skills: {result['skills']}")
    print(f"  address: {result['address']}")
    print(f"\n原始输出:\n{json.dumps(result, ensure_ascii=False, indent=2)}")

print()


# ═════════════════════════════════════════════════════════════════════════════
# 3. 数组输出 — 多条记录
# ═════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("3️⃣  数组输出 — 多条记录")
print("=" * 65)

system_prompt = """
从用户的商品描述中提取商品信息列表，以 JSON 数组格式输出。
每个商品包含：name（名称）、price（价格，数字）、category（分类）。

JSON 输出示例：
[
    {"name": "无线鼠标", "price": 99, "category": "电子产品"},
    {"name": "机械键盘", "price": 299, "category": "电子产品"}
]
"""

user_prompt = """
我们店里有三种商品：
1. 纯棉T恤，售价59元，属于服装类
2. 蓝牙耳机，售价199元，属于电子产品
3. 不锈钢保温杯，售价89元，属于日用品
"""

response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[  # type: ignore[arg-type]
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ],
    response_format={"type": "json_object"},  # type: ignore[arg-type]
    max_tokens=500,
    temperature=0.0,
)

content = response.choices[0].message.content
result = safe_json_parse(content)

if result:
    # 兼容外层是 {"items": [...]} 或直接数组
    items = result if isinstance(result, list) else result.get("items", result.get("products", [result]))
    print(f"\n共提取 {len(items)} 件商品:")
    for i, item in enumerate(items, 1):
        print(f"  {i}. {item['name']} — ¥{item['price']} ({item['category']})")
    print(f"\n原始输出:\n{json.dumps(result, ensure_ascii=False, indent=2)}")

print()


# ═════════════════════════════════════════════════════════════════════════════
# 4. 对比普通模式 vs JSON 模式
# ═════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("4️⃣  对比：普通模式 vs JSON 模式")
print("=" * 65)

prompt = (
    "分析以下句子的情感倾向（positive / negative / neutral）和主题。\n"
    "句子：今天天气真好，适合出去爬山！"
)

# ---------- 普通模式 ----------
print("  a) 普通模式（无 JSON 约束）")
resp_normal = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[  # type: ignore[arg-type]
        {"role": "system", "content": "你是一个情感分析助手。"},
        {"role": "user", "content": prompt},
    ],
    max_tokens=200,
    temperature=0.0,
)
print(f"     输出: {resp_normal.choices[0].message.content}")
print()

# ---------- JSON 模式 ----------
print("  b) JSON 模式（结构化解耦）")
system_json = """
你是一个情感分析助手。请以 JSON 格式输出分析结果，包含：
- sentiment: 情感分类（positive / negative / neutral）
- topic: 主题关键词
- confidence: 置信度（0~1）

JSON 输出示例：
{
    "sentiment": "positive",
    "topic": "天气",
    "confidence": 0.95
}
"""

resp_json = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[  # type: ignore[arg-type]
        {"role": "system", "content": system_json},
        {"role": "user", "content": prompt},
    ],
    response_format={"type": "json_object"},  # type: ignore[arg-type]
    max_tokens=200,
    temperature=0.0,
)

content = resp_json.choices[0].message.content
result = safe_json_parse(content)

if result:
    print(f"     sentiment:  {result['sentiment']}")
    print(f"     topic:      {result['topic']}")
    print(f"     confidence: {result['confidence']}")
    print(f"\n     原始 JSON:\n     {json.dumps(result, ensure_ascii=False)}")

print()


# ═════════════════════════════════════════════════════════════════════════════
# 5. 流式 JSON 输出（stream=True + json_object）
# ═════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("5️⃣  流式 JSON 输出")
print("=" * 65)

system_prompt = """
用户会提供书籍信息文本。请解析出 title（书名）、author（作者）、year（出版年份）并以 JSON 格式输出。

JSON 输出示例：
{
    "title": "三体",
    "author": "刘慈欣",
    "year": 2008
}
"""

user_prompt = "《百年孤独》是加西亚·马尔克斯在1967年出版的经典小说。"

stream = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[  # type: ignore[arg-type]
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ],
    response_format={"type": "json_object"},  # type: ignore[arg-type]
    max_tokens=300,
    temperature=0.0,
    stream=True,
)

# 流式拼接完整内容
full_content = ""
print("  流式接收中: ", end="", flush=True)
for chunk in stream:
    delta = chunk.choices[0].delta if chunk.choices else None
    if delta and delta.content:
        full_content += delta.content
        print(delta.content, end="", flush=True)

print("\n")
result = safe_json_parse(full_content)
if result:
    print(f"  解析结果:")
    print(f"    title:  {result['title']}")
    print(f"    author: {result['author']}")
    print(f"    year:   {result['year']}")

print()
print("=" * 65)
print("✅  JSON 模式演示完毕")
print("=" * 65)
