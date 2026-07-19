# DeepSeek API 文档深度讲解：多轮对话（Multi-Round Chat）

> 原文出处：https://api-docs.deepseek.com/zh-cn/guides/multi_round_chat

---

## 1. 一句话概览

这个网页讲的是：**怎么让 AI 记住你们俩之前聊过的话，而不是每说一句都像失忆一样从头开始。**

打个生活比方：你跟朋友聊天，你说"我昨天看了部电影"，朋友说"好看吗"，你说"特别好看"——朋友能听懂"特别好看"说的是那部电影，因为他记得上文。但 AI 默认是"金鱼记忆"，每次你发消息它都当你是个陌生人。多轮对话的技术，就是教你怎么把聊天记录每次都"喂"给 AI，让它表现得像个正常人一样能接上下文。

---

## 2. 全量专业术语扫盲

### 2.1 对话结构相关

- **多轮对话（Multi-Round Chat）**：你一言我一语的连续对话，后一句话依赖前一句话的语境。比如"它多少钱"——"它"指什么？必须看上文才知道。

- **单轮对话（Single-Round Chat）**：一问一答，互不依赖。比如"北京今天天气怎么样"——不需要任何上下文。

- **上下文（Context）**：对话的历史记录，也就是"之前说过的话"。AI 靠上下文来理解你当前这句话的意思。

- **消息（Message）**：对话中的每一条话，在 API 里用一个字典表示，包含角色和内容。

- **角色（Role）**：每条消息是谁说的。DeepSeek API 里有三种角色：
  - `system`：系统消息，相当于"给 AI 的设定说明书"，告诉它该扮演什么角色、遵守什么规则。比如"你是一个专业的法律顾问"。
  - `user`：用户消息，也就是"你"说的话。
  - `assistant`：助手消息，也就是 AI 之前回复的话。

### 2.2 API 调用相关

- **messages 数组**：你传给 API 的对话历史，是一个列表（数组），按时间顺序排列每条消息。AI 会"读"完整个列表，再生成下一条回复。

- **无状态（Stateless）**：这是最关键的概念！DeepSeek API（以及几乎所有大模型 API）是**无状态**的——服务器不会帮你保存聊天记录。每次你发请求，都必须把完整的对话历史重新发一遍。这跟你想的可能不一样：你以为 AI"记得"之前的话，其实是**你每次都把历史重新喂给它**。

- **上下文窗口（Context Window）**：模型一次能"读"的最大文本长度（按 token 算）。比如某个模型上下文窗口是 64K token，那你的 messages 数组总长度不能超过这个数，超过就要截断或压缩。

- **token**：文本的最小计量单位。中文大约 1 字 = 1~2 token，英文大约 1 词 = 1~1.5 token。你发的历史越长，消耗的 token 越多，花钱越多，速度也越慢。

### 2.3 编程相关

- **append（追加）**：往列表末尾加一个元素。多轮对话的核心操作就是：每次 AI 回复后，把它的回复 `append` 到 messages 列表里，下次请求时一起带上。

- **循环（Loop）**：反复执行一段代码。多轮对话通常用一个 `while True` 循环，不断接收用户输入、调 API、打印回复、把回复加回历史。

---

## 3. 核心知识与参数全解

### 3.1 多轮对话的本质：自己维护历史

很多人以为"多轮对话"是 API 的某个特殊功能，一开就自动记住上下文。**这是误解。** 真相是：

> API 本身是无状态的。所谓"多轮对话"，是**你的程序自己负责把历史消息累积起来，每次请求都带上完整历史**。

流程是这样的：

```
第 1 轮：
  你发：messages = [user: "我叫小明"]
  AI 回：assistant: "你好小明！"
  你把 AI 的回复加到 messages 里：
  messages = [user: "我叫小明", assistant: "你好小明！"]

第 2 轮：
  你发：messages = [user: "我叫小明", assistant: "你好小明！", user: "我叫什么？"]
  AI 回：assistant: "你叫小明"  ← 因为历史里有，所以它"记得"
  你再把这次对话加进去：
  messages = [user: "我叫小明", assistant: "你好小明！", user: "我叫什么？", assistant: "你叫小明"]
```

看到了吗？第 2 轮请求时，你把第 1 轮的完整对话都重新发了一遍。AI 之所以"记得"，是因为你告诉了它。

### 3.2 三种角色的使用规范

| 角色 | 何时用 | 位置 | 例子 |
|------|--------|------|------|
| `system` | 设定 AI 的人设、规则、风格 | 通常放 messages 数组**第一位**，且只放一条 | `{"role": "system", "content": "你是一个温柔的英语老师，用中英双语回答"}` |
| `user` | 用户每次的提问 | 按时间顺序穿插 | `{"role": "user", "content": "apple 怎么拼"}` |
| `assistant` | AI 之前的回复 | 按时间顺序穿插，紧跟在对应 user 之后 | `{"role": "assistant", "content": "apple 拼作 a-p-p-l-e"}` |

**重要规则**：messages 数组里的消息**必须按时间顺序排列**，且通常是 `user` 和 `assistant` 交替出现（一问一答）。乱序会导致 AI 困惑。

### 3.3 所有相关参数详解

#### 3.3.1 请求参数

| 参数名 | 数据类型 | 默认值 | 作用说明 |
|--------|---------|--------|---------|
| `model` | 字符串 | 无（必填） | 模型名，如 `deepseek-chat`、`deepseek-v4-pro` 等。多轮对话所有模型都支持。 |
| `messages` | 数组 | 无（必填） | 对话历史，按时间顺序排列。每条是 `{"role": "...", "content": "..."}`。**这是多轮对话的核心参数**——你把多少历史放进去，AI 就"记得"多少。 |
| `max_tokens` | 整数 | 无（建议填） | 本次回复的最大长度。注意：这只限制 AI **这一次**的回复长度，不限制你传入的历史长度。 |
| `temperature` | 浮点数 | 1.0 | 控制随机性。0 最确定（每次回答几乎一样），2 最随机（天马行空）。多轮对话一般用 0.7~1.0。 |
| `stream` | 布尔值 | false | 是否流式输出。多轮对话场景下建议 true，体验更好。 |
| `top_p` | 浮点数 | 1.0 | 另一种控制随机性的方式，和 temperature 二选一。一般不动。 |
| `frequency_penalty` | 浮点数 | 0 | 惩罚重复词。多轮对话里如果 AI 老重复同样的话，可以调高这个值（0~2）。 |
| `presence_penalty` | 浮点数 | 0 | 惩罚已出现的话题。想让 AI 多聊新话题可以调高。 |

#### 3.3.2 messages 数组里每条消息的字段

| 字段名 | 数据类型 | 是否必填 | 作用说明 |
|--------|---------|---------|---------|
| `role` | 字符串 | 必填 | 消息角色，只能是 `system`、`user`、`assistant` 三者之一。 |
| `content` | 字符串 | 必填 | 消息内容。可以是空字符串，但不能省略。 |

#### 3.3.3 返回参数（和多轮对话相关的）

| 字段名 | 数据类型 | 作用说明 |
|--------|---------|---------|
| `choices[0].message.content` | 字符串 | AI 本次回复的内容。**你必须把它加回 messages 数组**，下一轮才能保持上下文。 |
| `choices[0].message.role` | 字符串 | 始终是 `"assistant"`。加回历史时用这个。 |
| `usage.prompt_tokens` | 整数 | 本次请求输入的 token 数（**包含全部历史消息**）。轮数越多，这个数越大。 |
| `usage.completion_tokens` | 整数 | 本次回复的 token 数。 |
| `usage.total_tokens` | 整数 | prompt + completion。 |

### 3.4 多轮对话的成本陷阱

这是新手最容易忽略的点：**每多一轮，你就要把之前所有历史重新发一遍，token 消耗是累加的。**

举个例子，假设每轮对话大约 100 token：

- 第 1 轮：发送 100 token，AI 回 100 token → 消耗 200 token
- 第 2 轮：发送 200 token（第1轮+第2轮问题），AI 回 100 token → 消耗 300 token
- 第 3 轮：发送 300 token，AI 回 100 token → 消耗 400 token
- ……
- 第 10 轮：发送 1000 token，AI 回 100 token → 消耗 1100 token

10 轮下来，总消耗是 200+300+400+...+1100 = 6500 token，而不是你以为的 2000 token。**这就是为什么长对话会越来越慢、越来越贵。**

---

## 4. 代码示例剖析与环境配置指南

### 4.1 代码剖析

网页给出的多轮对话示例（整理后）：

```python
from openai import OpenAI

# 1. 创建客户端
client = OpenAI(
    api_key="<your api key>",
    base_url="https://api.deepseek.com",
)

# 2. 初始化对话历史
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the highest mountain in the world?"},
]

# 3. 第一轮请求
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=messages,
)

# 4. 把 AI 的回复加回历史
messages.append(response.choices[0].message)

# 5. 第二轮：用户继续问
messages.append(
    {"role": "user", "content": "What is the second?"}
)

# 6. 第二轮请求（带上了完整历史）
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=messages,
)

print(response.choices[0].message.content)
```

**逐块解释：**

- **第 1 块（创建客户端）**：和上一节一样，用 OpenAI 库连 DeepSeek 服务。

- **第 2 块（初始化历史）**：`messages` 是一个 Python 列表，先放一条 system 消息设定 AI 角色，再放第一条 user 消息。**为什么用列表？** 因为列表有顺序，能保留对话的时间线。

- **第 3 块（第一轮请求）**：把 messages 传给 API。API 返回的 `response.choices[0].message` 就是一个 `{"role": "assistant", "content": "..."}` 形式的字典（OpenAI 库会自动封装成对象，但可以当字典用）。

- **第 4 块（关键！把回复加回历史）**：`messages.append(response.choices[0].message)` 这一行是多轮对话的灵魂。如果不加这一步，下一轮请求时 AI 就"忘了"自己说过什么。**为什么这么写？** 因为 API 返回的 message 对象本身就符合 `{"role": "assistant", "content": "..."}` 格式，直接 append 进去就行，不用手动构造。

- **第 5 块（用户第二轮提问）**：再 append 一条 user 消息。注意"second"这个词——AI 能理解"second"指的是"第二高的山"，是因为历史里有"highest mountain"的上下文。

- **第 6 块（第二轮请求）**：再次调用 create，传入的 messages 现在包含 4 条：system、user1、assistant1、user2。AI 读完后回复，这次它能正确回答"第二高的山是 K2"。

### 4.2 从零跑通指南

#### 第一步：安装依赖

```bash
pip install openai
```

#### 第二步：获取并配置 API Key

参考"思考模式"文档的第四步，把密钥存到环境变量 `DEEPSEEK_API_KEY`。

#### 第三步：写一个能持续对话的程序

新建 `multi_round_chat.py`：

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

# 初始化对话历史，设定 AI 人设
messages = [
    {"role": "system", "content": "你是一个耐心的科普助手，用通俗易懂的语言回答问题。"}
]

print("多轮对话已启动，输入 'quit' 退出。\n")

while True:
    # 接收用户输入
    user_input = input("你: ")
    if user_input.strip().lower() in ("quit", "exit", "q"):
        print("再见！")
        break
    
    # 把用户的话加进历史
    messages.append({"role": "user", "content": user_input})
    
    # 调 API（带上完整历史）
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
    )
    
    # 取出 AI 回复
    ai_reply = response.choices[0].message.content
    print(f"AI: {ai_reply}\n")
    
    # 关键：把 AI 回复加回历史，下一轮才能接上下文
    messages.append({"role": "assistant", "content": ai_reply})
    
    # 可选：打印当前 token 消耗，感受成本增长
    print(f"[本次消耗: 输入 {response.usage.prompt_tokens} token, "
          f"输出 {response.usage.completion_tokens} token]\n")
```

#### 第四步：运行

```bash
python multi_round_chat.py
```

#### 第五步：预期输出

```
多轮对话已启动，输入 'quit' 退出。

你: 我叫张三
AI: 你好，张三！很高兴认识你。请问有什么想聊的吗？

[本次消耗: 输入 35 token, 输出 18 token]

你: 我叫什么？
AI: 你叫张三呀，你刚才告诉我的。

[本次消耗: 输入 60 token, 输出 15 token]

你: quit
再见！
```

注意看 `[本次消耗]` 那行：第二轮的输入 token 比第一轮多，因为历史变长了。这就是多轮对话的成本增长机制。

---

## 5. 生产环境视角升华（纯技术向）

### 5.1 适用场景

- **客服机器人**：用户描述问题、补充信息、追问解决方案，天然是多轮。
- **AI 助手/Agent**：执行复杂任务时需要多步交互，每步都依赖前文。
- **教育辅导**：学生做题、AI 讲解、学生追问、AI 进一步解释。
- **角色扮演/陪伴类应用**：长对话保持人设一致性。

### 5.2 避坑指南

- **上下文爆炸**：对话轮数多了，messages 数组会越来越长，最终超过模型的上下文窗口，导致报错或被截断。生产环境必须做上下文管理（见下文最佳实践）。

- **成本随轮数平方级增长**：如前文算的，10 轮对话的 token 消耗是单轮的 6 倍多。长对话场景必须做成本控制，否则一个用户聊一下午可能烧掉几十块。

- **历史消息污染**：如果用户中途说了错话或敏感内容，这些内容会一直留在历史里影响后续回复。需要做敏感词过滤和历史清理。

- **并发与一致性**：如果同一用户在多个设备同时对话，各自的 messages 历史可能不一致，导致 AI 表现"精神分裂"。需要统一在服务端管理会话状态。

- **assistant 消息格式必须规范**：如果你在历史里塞了格式不对的 assistant 消息（比如 content 是 None，或 role 拼错），API 会直接报错。从 tool_calls 场景拼接历史时尤其要小心。

- **不要把 tool 调用结果漏掉**：如果上一轮 AI 发起了 tool_calls，你必须在历史里同时保留：assistant 的 tool_calls 消息 + tool 角色的执行结果消息，否则下一轮 API 会报"missing tool result"错误。

### 5.3 最佳实践

**上下文管理策略（按优先级）：**

1. **滑动窗口截断**：保留最近 N 轮对话，丢弃更早的。简单有效，但会丢失早期重要信息。

```python
def trim_messages(messages, max_messages=20):
    """保留 system + 最近 N 条消息"""
    system_msgs = [m for m in messages if m["role"] == "system"]
    non_system = [m for m in messages if m["role"] != "system"]
    # 保留最近的 N 条
    recent = non_system[-max_messages:]
    return system_msgs + recent
```

2. **Token 预算截断**：按 token 数而非消息数截断，更精确。用 tiktoken 库估算 token 数。

3. **摘要压缩**：当历史超过阈值时，让另一个模型把早期对话总结成一段摘要，替换原始历史。成本高但信息保留好。

4. **混合策略**：近期对话保留原文 + 早期对话用摘要，是大多数生产系统的选择。

**会话存储架构：**

```python
import redis
import json

class SessionManager:
    """用 Redis 存储会话历史，支持多设备同步"""
    def __init__(self):
        self.redis = redis.Redis()
        self.ttl = 3600 * 24  # 会话保留 24 小时
    
    def get_messages(self, session_id: str):
        data = self.redis.get(f"session:{session_id}")
        return json.loads(data) if data else []
    
    def save_messages(self, session_id: str, messages: list):
        self.redis.setex(
            f"session:{session_id}",
            self.ttl,
            json.dumps(messages, ensure_ascii=False),
        )
```

**生产级多轮对话调用封装：**

```python
import os
from openai import OpenAI
import tiktoken

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

def count_tokens(messages, model="deepseek-chat"):
    """估算 messages 的 token 数"""
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")  # 近似估算
    total = 0
    for m in messages:
        total += len(enc.encode(m.get("content", "")))
        total += 4  # 每条消息的格式开销
    return total

def chat_with_history(session_id: str, user_input: str, max_context_tokens: int = 8000):
    # 1. 从存储加载历史
    messages = session_manager.get_messages(session_id)
    
    # 2. 追加新用户消息
    messages.append({"role": "user", "content": user_input})
    
    # 3. 按 token 预算截断（保留 system 和最近的消息）
    while count_tokens(messages) > max_context_tokens and len(messages) > 2:
        # 删除第二条（保留第一条 system）
        messages.pop(1)
    
    # 4. 调用 API
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
    )
    ai_reply = response.choices[0].message.content
    
    # 5. 追加 AI 回复到历史
    messages.append({"role": "assistant", "content": ai_reply})
    
    # 6. 持久化
    session_manager.save_messages(session_id, messages)
    
    return ai_reply, response.usage
```

**监控指标：**

- 平均对话轮数、平均上下文 token 数
- 因上下文超限被截断的请求比例
- 单会话 token 消耗 P50/P95/P99
- 会话存储的命中率与延迟
