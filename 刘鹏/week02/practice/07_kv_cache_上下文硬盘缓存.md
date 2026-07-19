# DeepSeek API 文档深度讲解：上下文硬盘缓存（Context Caching on Disk）

> 原文出处：https://api-docs.deepseek.com/zh-cn/guides/kv_cache

---

## 1. 一句话概览

这个网页讲的是：**DeepSeek 会自动把你重复发送的"前情提要"缓存到硬盘上，下次再发同样的内容时直接复用，不重新算，从而给你省钱、提速。**

打个生活比方：你每天给老板汇报工作，每次都要先念一遍"公司背景、项目简介、团队介绍"这套固定开场白，然后再说今天的进展。老板每次听开场白都要花 10 分钟重新理解。后来老板说"这套开场白我记住了，以后你直接说进展就行"——这就是缓存。DeepSeek 的硬盘缓存就是这个机制：重复的"开场白"（前缀）它记住了，下次直接用，省时省钱。

---

## 2. 全量专业术语扫盲

### 2.1 缓存核心概念

- **缓存（Cache）**：把重复使用的数据临时存起来，下次要用时直接取，不重新计算。就像你把常用电话号码存手机通讯录，下次直接拨，不用每次都查。

- **上下文缓存（Context Caching）**：特指把对话的上下文（前情提要）缓存起来。DeepSeek 把这个功能叫"上下文硬盘缓存"。

- **KV Cache（Key-Value Cache）**：这是大模型内部的术语。模型在处理文本时，会把每一层的中间计算结果（叫 Key 和 Value）存下来，避免重复计算。DeepSeek 把这些 KV 缓存持久化到硬盘上，跨请求复用。你不用深究 KV 是什么，把它理解成"模型算到一半的草稿"就行。

- **硬盘缓存（Disk Cache）**：缓存放内存里重启就没了，放硬盘上能持久。DeepSeek 把 KV Cache 写到硬盘，所以叫"硬盘缓存"。好处是即使服务器重启，缓存还在。

### 2.2 命中与计费相关

- **缓存命中（Cache Hit）**：你发的内容，前面有一部分和之前发过的某次完全一样，那部分就直接用缓存，不重新算。命中越多，省越多。

- **缓存未命中（Cache Miss）**：你发的内容和之前的不一样，只能重新算。未命中的部分正常计费。

- **prompt_cache_hit_tokens（缓存命中 token 数）**：你这次请求里，有多少 token 命中了缓存（白嫖成功）。这个数越大，省得越多。

- **prompt_cache_miss_tokens（缓存未命中 token 数）**：你这次请求里，有多少 token 没命中缓存（得重新算）。这个数正常计费。

- **缓存命中单价**：命中缓存的 token，单价**大幅低于**正常输入 token。这是 DeepSeek 给你的折扣。

### 2.3 前缀与匹配相关

- **前缀（Prefix）**：缓存是按"前缀匹配"工作的。也就是说，缓存命中的是从**开头**开始连续相同的部分。一旦遇到第一个不同的 token，后面就全部未命中。

- **前缀匹配（Prefix Matching）**：就像查字典——你查"应用"，字典里有"应用程序"，那"应用"这部分命中，"程序"是新内容。但如果你查"程序应用"，前面"程序"和字典里的"应用"开头不同，就完全不命中。

- **自动缓存（Automatic Caching）**：DeepSeek 的缓存是**全自动**的，你不需要改任何代码、加任何参数。只要你的请求前缀和之前某次一样，就自动命中。

### 2.4 编程与监控相关

- **usage 字段**：API 返回的 token 使用统计。除了常规的 prompt_tokens、completion_tokens，还有 prompt_cache_hit_tokens 和 prompt_cache_miss_tokens。

- **缓存命中率**：`prompt_cache_hit_tokens / prompt_tokens`。这个比例越高，省得越多。生产环境要监控这个指标。

- **TTL（Time To Live，存活时间）**：缓存的保留时长。DeepSeek 的缓存有 TTL，超过一段时间不用就自动清理。文档说目前缓存至少保留一段时间（具体时长以官网为准）。

---

## 3. 核心知识与参数全解

### 3.1 缓存的工作原理（必懂）

这是整个功能的核心，务必理解：

```
第一次请求：
messages = [
    {role: system, content: "你是一个专业的法律顾问...(5000字的长设定)"},
    {role: user, content: "什么是不可抗力？"}
]
→ DeepSeek 处理这 5000 字设定 + 问题，生成回答
→ 同时，把这 5000 字设定的 KV Cache 存到硬盘
→ 你付 5000 token 的正常输入费

第二次请求（用同样的 system）：
messages = [
    {role: system, content: "你是一个专业的法律顾问...(5000字的长设定)"},  ← 命中缓存！
    {role: user, content: "合同违约怎么处理？"}  ← 未命中
]
→ DeepSeek 发现前 5000 字和上次一样，直接用硬盘上的缓存
→ 只有"合同违约怎么处理？"这部分要重新算
→ 你付：5000 token 的缓存命中价（便宜很多）+ 几十个 token 的正常价
```

**关键认知**：
1. 缓存是**自动**的，零代码改动。
2. 命中是**前缀匹配**，从开头连续相同才算。
3. 命中部分**便宜很多**（通常是正常价的零头）。
4. 缓存有 **TTL**，长时间不用会失效。

### 3.2 所有相关参数与字段详解

#### 3.2.1 请求侧（你不需要额外参数）

缓存是自动的，你正常调 API 即可。但有几个**隐含规则**影响命中：

| 影响因素 | 说明 |
|---------|------|
| `model` | 必须和之前一致。不同模型的缓存不通用。 |
| `messages` 前缀 | 从第一条消息开始，必须和之前完全一致（逐 token 相同）。 |
| `temperature` 等参数 | **不影响**缓存命中（缓存只看输入，不看采样参数）。 |
| `tools` | 如果用了工具，tools 定义也是前缀的一部分，必须一致。 |
| `system` 消息 | 通常放最前面，是最容易被缓存命中的部分。 |

#### 3.2.2 返回侧字段

| 字段名 | 数据类型 | 作用说明 |
|--------|---------|---------|
| `usage.prompt_tokens` | 整数 | 总输入 token 数 = hit + miss。 |
| `usage.completion_tokens` | 整数 | 输出 token 数（正常计费，不涉及缓存）。 |
| `usage.prompt_cache_hit_tokens` | 整数 | **命中缓存的 token 数**。这部分按缓存价计费（便宜）。 |
| `usage.prompt_cache_miss_tokens` | 整数 | **未命中的 token 数**。这部分按正常输入价计费。 |

**计费公式**（理解这个就懂了怎么省钱）：

```
输入费用 = prompt_cache_hit_tokens × 缓存命中单价
         + prompt_cache_miss_tokens × 正常输入单价
         + completion_tokens × 输出单价
```

### 3.3 如何最大化缓存命中

这是工程实践的核心。原则：**把不变的内容放前面，变化的内容放后面。**

#### 错误做法（命中率低）

```python
# 每次都把用户输入放最前面，system 放后面
messages = [
    {"role": "user", "content": f"用户{user_id}问：{question}"},  # 每次都变
    {"role": "system", "content": "你是一个专业的法律顾问...(5000字)"},  # 不变但放后面
]
# 前缀一变就全不命中，5000字白付
```

#### 正确做法（命中率高）

```python
# system 放最前，固定的 few-shot 示例放中间，用户输入放最后
messages = [
    {"role": "system", "content": "你是一个专业的法律顾问...(5000字)"},  # 命中！
    {"role": "user", "content": "示例问题1"},  # few-shot，命中
    {"role": "assistant", "content": "示例回答1"},  # 命中
    {"role": "user", "content": question},  # 只有这里未命中
]
```

### 3.4 缓存失效的场景

以下情况缓存会失效或不命中：

1. **前缀变了**：哪怕改一个字，从那以后全部未命中。
2. **TTL 过期**：长时间不请求，缓存被清理。
3. **模型升级**：模型版本变了，旧缓存作废。
4. **流量高峰清理**：极端情况下，服务器可能清理冷缓存腾空间。

---

## 4. 代码示例剖析与环境配置指南

### 4.1 代码剖析

网页强调缓存是自动的，不需要特殊代码。但你可以**观察**缓存命中情况：

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

# 一个很长的固定 system prompt（模拟实际项目中的复杂设定）
LONG_SYSTEM_PROMPT = """
你是一个专业的法律顾问助手。你的职责是：
（这里省略 5000 字的详细设定、法律知识库、回答规范等）
""" * 50  # 故意弄长一点

# 第一次请求
messages = [
    {"role": "system", "content": LONG_SYSTEM_PROMPT},
    {"role": "user", "content": "什么是不可抗力？"},
]

response1 = client.chat.completions.create(
    model="deepseek-chat",
    messages=messages,
)

print("=== 第一次请求 ===")
print(f"prompt_tokens: {response1.usage.prompt_tokens}")
print(f"prompt_cache_hit_tokens: {response1.usage.prompt_cache_hit_tokens}")  # 应该是 0 或很少
print(f"prompt_cache_miss_tokens: {response1.usage.prompt_cache_miss_tokens}")  # 几乎全部

# 第二次请求，system 不变，只换 user 问题
messages2 = [
    {"role": "system", "content": LONG_SYSTEM_PROMPT},  # 和第一次完全一样
    {"role": "user", "content": "合同违约怎么处理？"},  # 换了问题
]

response2 = client.chat.completions.create(
    model="deepseek-chat",
    messages=messages2,
)

print("\n=== 第二次请求 ===")
print(f"prompt_tokens: {response2.usage.prompt_tokens}")
print(f"prompt_cache_hit_tokens: {response2.usage.prompt_cache_hit_tokens}")  # 应该很大（system 命中）
print(f"prompt_cache_miss_tokens: {response2.usage.prompt_cache_miss_tokens}")  # 只有新问题部分

# 计算命中率
hit_rate = response2.usage.prompt_cache_hit_tokens / response2.usage.prompt_tokens
print(f"\n缓存命中率: {hit_rate:.1%}")
```

**逐块解释：**

- **LONG_SYSTEM_PROMPT**：模拟实际项目里很长的 system 设定。缓存的价值就在于这种长前缀。

- **第一次请求**：第一次发，缓存还没建立，所以 `prompt_cache_hit_tokens` 为 0，全部 miss。但这次请求会**建立**缓存。

- **第二次请求**：system 完全一样，所以 system 部分命中缓存。只有新的 user 问题部分 miss。你会看到 `prompt_cache_hit_tokens` 很大。

- **命中率计算**：`hit / total`。这个比例越高，省得越多。生产环境要持续监控。

### 4.2 从零跑通指南

#### 第一步：安装依赖

```bash
pip install openai python-dotenv
```

#### 第二步：配置 API Key

```bash
export DEEPSEEK_API_KEY="sk-你的密钥"
```

#### 第三步：运行

```bash
python kv_cache_test.py
```

#### 第四步：预期输出

```
=== 第一次请求 ===
prompt_tokens: 5020
prompt_cache_hit_tokens: 0
prompt_cache_miss_tokens: 5020

=== 第二次请求 ===
prompt_tokens: 5015
prompt_cache_hit_tokens: 5000
prompt_cache_miss_tokens: 15

缓存命中率: 99.7%
```

（具体数字会变，但第二次的 hit 应该远大于 miss）

#### 验证省了多少钱

假设：
- 正常输入价：1元/百万 token
- 缓存命中价：0.1元/百万 token（具体以官网为准）

第一次：5020 × 1 = 0.00502 元
第二次：5000 × 0.1 + 15 × 1 = 0.0005 + 0.000015 = 0.000515 元

第二次省了约 90%！这就是缓存的价值。

#### 常见问题

- **第二次还是没命中**：可能是 TTL 过期（两次请求间隔太久），或 system 内容有细微差别（多一个空格都不行）。
- **命中率不稳定**：流量高峰时缓存可能被清理。生产环境要监控命中率波动。
- **不同模型不通用**：用 deepseek-chat 建的缓存，换 deepseek-reasoner 就不命中。

---

## 5. 生产环境视角升华（纯技术向）

### 5.1 适用场景

- **长 system prompt**：复杂的角色设定、知识库、规则文档放在 system 里，每次请求都带，缓存命中后大幅降本。
- **Few-shot learning**：固定的示例（user-assistant 对话对）放前面，每次只换最后的问题。
- **RAG 系统**：检索到的文档作为固定上下文，多次查询同一文档时命中缓存。
- **多轮对话**：对话历史每次累积，前缀部分命中缓存，只有新增部分计费。
- **批量处理**：同一模板处理大量数据，模板部分命中。
- **Agent 系统**：工具定义（tools）固定不变，每次都带，命中缓存。

### 5.2 避坑指南

- **前缀顺序敏感**：messages 的顺序一变，缓存就失效。务必把固定内容放最前，变化内容放最后。

- **微小改动导致全失效**：多一个空格、换一个标点、改一个字，从那以后全部未命中。团队协作时要规范 prompt 管理，避免无意改动。

- **TTL 失效**：低频访问的场景，缓存可能在下次请求前就过期。对于重要缓存，可以用定时任务"保活"（定期发一个请求维持缓存）。

- **缓存命中率波动**：服务器负载高时可能清理缓存。别假设缓存一定命中，成本估算要按最坏情况留余量。

- **多模型混用**：不同模型的缓存独立。如果系统里混用多个模型，每个模型的缓存都要单独建立。

- **动态内容污染前缀**：如果在 system 里塞了动态内容（如当前时间、用户ID），前缀就每次都变，缓存全失效。动态内容要放最后。

- **监控盲区**：不监控 prompt_cache_hit_tokens，就不知道缓存有没有生效、省了多少钱。必须接入监控。

- **缓存击穿**：大量请求同时 miss（如缓存集体过期后的第一波请求），可能造成延迟飙升。要做限流和预热。

### 5.3 最佳实践

**Prompt 结构设计原则：**

```
[最固定] system prompt（角色设定、规则）
   ↓
[较固定] few-shot 示例
   ↓
[较固定] 检索到的文档（RAG）
   ↓
[半固定] 对话历史（多轮）
   ↓
[最变化] 当前用户输入
```

越固定的越靠前，越变化的越靠后。

**生产级缓存监控：**

```python
import os
import logging
from openai import OpenAI
from dataclasses import dataclass
from datetime import datetime

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

logger = logging.getLogger(__name__)

@dataclass
class CacheStats:
    total_prompt_tokens: int = 0
    total_hit_tokens: int = 0
    total_miss_tokens: int = 0
    request_count: int = 0
    
    def record(self, usage):
        self.total_prompt_tokens += usage.prompt_tokens
        self.total_hit_tokens += usage.prompt_cache_hit_tokens
        self.total_miss_tokens += usage.prompt_cache_miss_tokens
        self.request_count += 1
    
    @property
    def hit_rate(self):
        return self.total_hit_tokens / self.total_prompt_tokens if self.total_prompt_tokens else 0

cache_stats = CacheStats()

def chat_with_cache_tracking(messages, **kwargs):
    """带缓存监控的聊天调用"""
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        **kwargs,
    )
    
    usage = response.usage
    cache_stats.record(usage)
    
    hit_rate = usage.prompt_cache_hit_tokens / usage.prompt_tokens if usage.prompt_tokens else 0
    logger.info(
        f"Cache: hit={usage.prompt_cache_hit_tokens} "
        f"miss={usage.prompt_cache_miss_tokens} "
        f"rate={hit_rate:.1%}"
    )
    
    # 命中率过低告警
    if usage.prompt_tokens > 1000 and hit_rate < 0.3:
        logger.warning(
            f"Low cache hit rate ({hit_rate:.1%}) for prompt with "
            f"{usage.prompt_tokens} tokens. Consider reordering messages."
        )
    
    return response

# Prompt 模板管理（避免无意改动导致缓存失效）
class PromptManager:
    """集中管理 prompt 模板，避免团队成员随意改动导致缓存失效"""
    _templates = {}
    
    @classmethod
    def register(cls, name: str, template: str):
        cls._templates[name] = template
    
    @classmethod
    def get(cls, name: str) -> str:
        if name not in cls._templates:
            raise KeyError(f"Prompt template '{name}' not registered")
        return cls._templates[name]

# 注册一次，全局复用
PromptManager.register("legal_advisor", "你是一个专业的法律顾问..." * 50)

# 使用
messages = [
    {"role": "system", "content": PromptManager.get("legal_advisor")},  # 固定，命中缓存
    {"role": "user", "content": user_question},  # 变化
]
response = chat_with_cache_tracking(messages)
```

**缓存预热（对关键场景）：**

```python
import threading
import time

def warmup_cache(system_prompt: str, interval: int = 300):
    """定期发请求维持缓存，防止 TTL 过期"""
    while True:
        try:
            client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "ping"},
                ],
                max_tokens=1,
            )
            logger.debug("Cache warmed up")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")
        time.sleep(interval)

# 对重要的长 system prompt 启动预热
# threading.Thread(target=warmup_cache, args=(LONG_SYSTEM_PROMPT,), daemon=True).start()
```

**监控指标：**

- 整体缓存命中率（应 >70% 对于长 prompt 场景）
- 各 prompt 模板的命中率分布
- 缓存命中 token 总量 / 未命中 token 总量
- 缓存节省的成本（对比无缓存的费用）
- 命中率随时间的波动（识别 TTL 失效或服务器清理）
- 低命中率请求的告警和归因
