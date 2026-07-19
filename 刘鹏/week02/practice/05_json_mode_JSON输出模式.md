# DeepSeek API 文档深度讲解：JSON 输出模式（JSON Mode）

> 原文出处：https://api-docs.deepseek.com/zh-cn/guides/json_mode

---

## 1. 一句话概览

这个网页讲的是：**怎么强制 AI 把回答写成电脑能直接读懂的"表格格式"（JSON），而不是说人话，这样你的程序就能直接拿来做后续处理。**

打个生活比方：你问同事"今天天气怎么样"，普通回答是"今天挺热的，记得带伞"——这是说给人听的。但如果你要做天气预报 App，你希望同事填一张表：`{"温度": "35度", "天气": "雷阵雨", "建议": "带伞"}`——这是说给电脑听的。JSON 模式就是逼着 AI 按这种"填表"的方式回答，省得你还要从人话里抠信息。

---

## 2. 全量专业术语扫盲

### 2.1 JSON 相关

- **JSON（JavaScript Object Notation，JavaScript 对象表示法）**：一种通用的数据格式，长这样 `{"key": "value"}`。几乎所有编程语言都能读写它。你可以把它理解成"电脑版的填表格式"——左边是字段名，右边是值，用大括号包起来。

- **JSON 对象（JSON Object）**：用 `{}` 包起来的键值对集合。比如 `{"name": "张三", "age": 32}`。

- **JSON 数组（JSON Array）**：用 `[]` 包起来的有序列表。比如 `[1, 2, 3]` 或 `[{"name": "张三"}, {"name": "李四"}]`。

- **键值对（Key-Value Pair）**：JSON 的基本组成单位。`"name": "张三"` 就是一个键值对，`name` 是键（字段名），`"张三"` 是值。

- **解析（Parse）**：把 JSON 字符串转成程序里的数据结构（如 Python 的字典）。Python 里用 `json.loads()`。

- **序列化（Serialize）**：反过来，把程序里的数据结构转成 JSON 字符串。Python 里用 `json.dumps()`。

### 2.2 API 与参数相关

- **JSON Mode（JSON 模式）**：DeepSeek API 的一个开关，开启后 AI 的回复**保证**是合法的 JSON 字符串，可以直接被程序解析。

- **response_format（响应格式）**：API 的一个参数，用来指定 AI 回复的格式。设成 `{"type": "json_object"}` 就是开启 JSON 模式。

- **json_object（JSON 对象类型）**：response_format 的一种取值，表示"回复必须是一个 JSON 对象"。

- **Schema（模式/结构定义）**：JSON 的"模板"或"字段说明书"，规定 JSON 里该有哪些字段、每个字段是什么类型。比如 `{"name": string, "age": number}` 就是一个 schema。

### 2.3 编程相关

- **结构化输出（Structured Output）**：让 AI 按预定义的结构（schema）输出，而不是自由文本。JSON 模式是实现结构化输出的一种手段。

- **正则表达式（Regex）**：一种文本匹配工具。在 JSON 模式出现之前，人们常用正则从 AI 的自由文本回复里"抠"数据，又脆弱又难写。JSON 模式就是为了取代这种做法。

---

## 3. 核心知识与参数全解

### 3.1 为什么需要 JSON 模式？

**痛点**：假设你让 AI 做情感分析，你希望拿到 `{"sentiment": "正面", "score": 0.9}` 这样的结构化结果。但 AI 可能这样回你：

```
好的，我来帮你分析这段文本的情感。
分析结果如下：
情感倾向：正面
置信度：0.9
希望对你有帮助！
```

这段话人能看懂，但程序要从中提取"正面"和"0.9"非常麻烦——你得写正则匹配"情感倾向："后面的内容，万一 AI 换个说法"情感："就失效了。

**JSON 模式的解法**：开启后，AI 直接输出：

```json
{"sentiment": "正面", "score": 0.9}
```

程序一行 `json.loads()` 就拿到字典，直接用 `data["sentiment"]` 取值。稳定、可靠、零解析成本。

### 3.2 JSON 模式的工作机制

开启 JSON 模式后，DeepSeek 在模型层面做了约束：**生成过程中只允许输出合法 JSON 字符的 token**。也就是说，模型在"想说人话"的时候，会被强制走 JSON 语法路径。这比单纯在 prompt 里写"请输出 JSON"可靠得多——后者模型可能"忘记"或"加戏"。

### 3.3 所有相关参数详解

#### 3.3.1 请求参数

| 参数名 | 数据类型 | 默认值 | 作用说明 |
|--------|---------|--------|---------|
| `model` | 字符串 | 无（必填） | 模型名，如 `deepseek-chat`。 |
| `messages` | 数组 | 无（必填） | 对话历史。**JSON 模式下，messages 里必须包含 "json" 这个词**（在 user 或 system 消息里），否则 API 会报错。这是 DeepSeek 的强制要求，防止你开了 JSON 模式但没告诉模型要干嘛。 |
| `response_format` | 字典 | 无 | **核心参数**。设为 `{"type": "json_object"}` 开启 JSON 模式。不传或传 `{"type": "text"}` 就是普通文本模式。 |
| `response_format.type` | 字符串 | "text" | 取值 `"json_object"` 开启 JSON 模式；`"text"` 普通文本。 |
| `max_tokens` | 整数 | 无（建议填） | 回复最大长度。JSON 模式下要给够，否则 JSON 可能被截断导致解析失败。 |
| `temperature` | 浮点数 | 1.0 | 随机性。JSON 模式建议低温度（0~0.3），保证输出稳定。 |
| `stream` | 布尔值 | false | 是否流式。JSON 模式可以流式，但要注意：流式时每个 chunk 只是 JSON 片段，要全部拼起来才能解析。 |

#### 3.3.2 返回参数

| 字段名 | 数据类型 | 作用说明 |
|--------|---------|---------|
| `choices[0].message.content` | 字符串 | AI 的回复，**是一个 JSON 字符串**（不是字典，是字符串形式的 JSON）。你需要自己 `json.loads()` 转成字典。 |
| `choices[0].finish_reason` | 字符串 | `stop` = 正常结束；`length` = 被 max_tokens 截断（JSON 可能不完整）。 |

#### 3.3.3 关键约束

1. **messages 必须提到 "json"**：这是硬性要求。如果你开了 `response_format={"type": "json_object"}` 但 messages 里一个 "json" 都没有，API 会直接报错。**为什么？** DeepSeek 这么设计是为了防止误用——你必须明确告诉模型"我要 JSON"，模型才知道要按什么结构输出。

2. **输出是 JSON 字符串，不是字典**：API 返回的 `content` 是一个字符串，内容是 JSON 文本。你必须用 `json.loads()` 解析后才能当字典用。新手常踩这个坑。

3. **不保证字段 schema**：JSON 模式只保证输出是**合法的 JSON**，但不保证一定有你想要的字段。比如你想要 `{"name": ..., "age": ...}`，模型可能给你 `{"username": ..., "user_age": ...}`——字段名变了。要严格约束字段，需要在 prompt 里明确说明，或用更高级的 Structured Output（schema 约束）功能。

4. **max_tokens 要给够**：JSON 比纯文本多很多符号（`{}[]"":,`），token 消耗更大。给小了容易被截断，截断的 JSON 无法解析。

### 3.4 JSON 模式 vs Prompt 引导 vs Structured Output

| 方式 | 可靠性 | 实现难度 | 字段约束 |
|------|--------|---------|---------|
| Prompt 里写"请输出 JSON" | 低（模型可能不遵守） | 简单 | 无 |
| **JSON 模式** | **高（强制合法 JSON）** | **简单** | **无（只保证合法，不保证字段）** |
| Structured Output（schema 约束） | 最高（强制字段） | 中等 | 有 |

---

## 4. 代码示例剖析与环境配置指南

### 4.1 代码剖析

网页给出的 JSON 模式示例（整理后）：

```python
from openai import OpenAI

# 1. 创建客户端
client = OpenAI(
    api_key="<your api key>",
    base_url="https://api.deepseek.com",
)

# 2. 构造 messages，注意 prompt 里要提到 "json"
messages = [
    {"role": "user", "content": "给我一个示例 JSON，包含一个人的姓名、年龄和职业。请用 JSON 格式输出。"}
]

# 3. 调用，开启 JSON 模式
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=messages,
    response_format={"type": "json_object"},  # ← 开启 JSON 模式
)

# 4. 解析返回的 JSON 字符串
import json
result = json.loads(response.choices[0].message.content)
print(result)
print(type(result))  # <class 'dict'>
```

**逐块解释：**

- **第 1 块（创建客户端）**：标准操作，普通 base_url 即可。

- **第 2 块（构造 messages）**：**关键**——content 里必须有 "json" 这个词。这里写了"请用 JSON 格式输出"，满足要求。**为什么强制？** 见上文约束说明。

- **第 3 块（调用 API）**：`response_format={"type": "json_object"}` 是开启 JSON 模式的开关。这一行让 DeepSeek 在生成时强制走 JSON 语法路径。

- **第 4 块（解析结果）**：`response.choices[0].message.content` 拿到的是**字符串**（内容是 JSON 文本），用 `json.loads()` 转成 Python 字典。之后就能用 `result["姓名"]` 这样的方式取值了。

### 4.2 从零跑通指南

#### 第一步：安装依赖

```bash
pip install openai
```

#### 第二步：配置 API Key

参考前面文档，把密钥存到环境变量 `DEEPSEEK_API_KEY`。

#### 第三步：写代码

新建 `json_mode_test.py`：

```python
import os
import json
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

# 业务场景：情感分析，要求结构化输出
user_input = "这家餐厅的服务态度真好，菜也好吃，就是有点贵。"

messages = [
    {
        "role": "system",
        "content": "你是情感分析助手。请用 JSON 格式输出分析结果，包含字段：sentiment（情感：正面/负面/中性）、score（置信度0-1）、keywords（关键词数组）。"
    },
    {
        "role": "user",
        "content": f"分析这段文本的情感：{user_input}"
    }
]

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=messages,
    response_format={"type": "json_object"},
    max_tokens=512,
    temperature=0.1,  # 低温度保证稳定
)

# 解析 JSON
content = response.choices[0].message.content
print("=== 原始返回（字符串）===")
print(content)
print(type(content))  # str

print("\n=== 解析后（字典）===")
data = json.loads(content)
print(data)
print(type(data))  # dict

print("\n=== 取值使用 ===")
print(f"情感倾向: {data['sentiment']}")
print(f"置信度: {data['score']}")
print(f"关键词: {data['keywords']}")
```

#### 第四步：运行

```bash
python json_mode_test.py
```

#### 第五步：预期输出

```
=== 原始返回（字符串）===
{"sentiment": "正面", "score": 0.85, "keywords": ["服务态度好", "菜好吃", "贵"]}
<class 'str'>

=== 解析后（字典）===
{'sentiment': '正面', 'score': 0.85, 'keywords': ['服务态度好', '菜好吃', '贵']}
<class 'dict'>

=== 取值使用 ===
情感倾向: 正面
置信度: 0.85
关键词: ['服务态度好', '菜好吃', '贵']
```

#### 常见错误排查

- **报错 "messages must contain the word 'json'"**：你的 messages 里没出现 "json" 这个词。在 user 或 system 消息里加上"请用 JSON 格式输出"即可。
- **json.loads() 报错 JSONDecodeError**：可能是 max_tokens 太小，JSON 被截断。调大 max_tokens。或者检查 finish_reason 是不是 `length`。
- **字段名和预期不符**：JSON 模式只保证合法 JSON，不保证字段名。在 prompt 里明确写"必须包含字段：xxx、yyy"。
- **content 是 None**：检查 finish_reason，可能是被内容审核拦截了。

---

## 5. 生产环境视角升华（纯技术向）

### 5.1 适用场景

- **信息抽取**：从非结构化文本（新闻、评论、简历）中提取结构化字段（实体、属性、关系），存入数据库。
- **分类任务**：情感分析、意图识别、垃圾内容检测，输出 `{"label": "...", "confidence": ...}`。
- **数据增强/生成**：批量生成测试数据、mock 数据，要求结构化。
- **Agent 间通信**：多个 AI Agent 协作时，用 JSON 作为消息格式，便于程序解析和路由。
- **表单填充**：用户自然语言描述需求，AI 转成结构化表单数据，直接入库或触发流程。
- **RAG 管道**：从文档抽取 QA 对、摘要、元数据，结构化存储供检索。

### 5.2 避坑指南

- **字段约束不可靠**：JSON 模式只保证合法 JSON，不保证字段。生产环境要：
  1. 在 prompt 里用强约束语言（"必须包含且仅包含以下字段：..."）。
  2. 解析后做 schema 校验（用 Pydantic、jsonschema 库），字段缺失或类型不符时重试或降级。
  3. 考虑用更高级的 Structured Output 功能（如果 DeepSeek 支持 schema 约束）。

- **截断风险**：JSON 被截断后无法解析。生产环境要：
  1. max_tokens 给足（按业务最大输出估算 ×1.5 安全系数）。
  2. 检查 finish_reason，是 `length` 就重试或报错。
  3. 实现容错解析：尝试补全缺失的 `}`、`]`、`"` 后再解析。

- **嵌套深度限制**：过深的 JSON 嵌套模型容易出错。尽量保持扁平结构，嵌套不超过 3 层。

- **数组长度不可控**：模型可能输出 0 个或 100 个数组元素。在 prompt 里限定"最多 5 个"，或在程序侧截断。

- **数值类型混淆**：模型可能把数字输出成字符串 `"0.9"` 而不是数字 `0.9`。解析后做类型转换和校验。

- **空值处理**：模型可能输出 `null` 或空字符串。程序侧要处理 None/空值，别让下游崩溃。

- **成本敏感**：JSON 比纯文本多符号，token 消耗高 20%~50%。大批量任务要算清成本。

- **多语言字段名**：用中文字段名可能不稳定（编码、转义问题）。生产环境建议用英文蛇形命名（`user_name`、`sentiment_score`）。

### 5.3 最佳实践

**用 Pydantic 做 schema 约束和校验：**

```python
import os
import json
from pydantic import BaseModel, Field, ValidationError
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

# 1. 定义期望的输出 schema
class SentimentResult(BaseModel):
    sentiment: str = Field(..., description="情感倾向：正面/负面/中性")
    score: float = Field(..., ge=0, le=1, description="置信度，0到1之间")
    keywords: list[str] = Field(default_factory=list, description="关键词列表")
    summary: str = Field(..., description="一句话总结")

# 2. 把 schema 告诉模型
schema_hint = """请输出 JSON，必须包含以下字段：
- sentiment: 字符串，取值"正面"/"负面"/"中性"
- score: 数字，0到1之间
- keywords: 字符串数组
- summary: 字符串，一句话总结
不要输出任何其他字段。"""

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
def analyze_sentiment(text: str) -> SentimentResult:
    """带重试和 schema 校验的情感分析"""
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": schema_hint},
            {"role": "user", "content": f"分析：{text}"},
        ],
        response_format={"type": "json_object"},
        max_tokens=512,
        temperature=0.1,
    )
    
    content = response.choices[0].message.content
    
    # 检查是否被截断
    if response.choices[0].finish_reason == "length":
        raise ValueError("Response truncated, increase max_tokens")
    
    # 解析 JSON
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}\nContent: {content}")
    
    # Pydantic 校验 schema
    try:
        return SentimentResult(**data)
    except ValidationError as e:
        raise ValueError(f"Schema validation failed: {e}\nData: {data}")

# 使用
try:
    result = analyze_sentiment("这家餐厅服务好，菜好吃，就是贵。")
    print(result.model_dump_json(indent=2))
    # result 是强类型的 SentimentResult 对象，可以直接用 result.sentiment 等
except Exception as e:
    print(f"分析失败: {e}")
```

**容错解析（处理截断的 JSON）：**

```python
def safe_json_parse(text: str):
    """尝试解析 JSON，失败时尝试修复"""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # 尝试补全缺失的引号和括号
    repaired = text
    # 统计未闭合的符号
    for opener, closer in [("{", "}"), ("[", "]"), ('"', '"')]:
        diff = repaired.count(opener) - repaired.count(closer)
        if opener == '"':
            if diff % 2 == 1:
                repaired += '"'
        else:
            repaired += closer * diff
    
    try:
        return json.loads(repaired)
    except json.JSONDecodeError as e:
        raise ValueError(f"Cannot parse JSON even after repair: {e}")
```

**监控指标：**

- JSON 解析成功率（应 >99%）
- Schema 校验通过率
- 因 `length` 截断的请求比例
- 平均输出 token 数（监控成本）
- 不同 prompt 模板的字段准确率对比
