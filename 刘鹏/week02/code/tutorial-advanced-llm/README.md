# tutorial-advanced-llm — 生产级多模型多格式 LLM 高级实战

> 面向生产环境的大模型调用最佳实践，涵盖多环境配置、多格式适配、思考模式、工具调用、上下文管理五大核心主题。

---

## 目录结构

```
tutorial-advanced-llm/
├── advanced_llm/                  # 核心包（可 pip install）
│   ├── __init__.py                # 包入口，统一导出
│   ├── config.py                  # 多环境 + 多模型配置管理
│   ├── models.py                  # 数据模型（ChatResult, StreamChunk 等）
│   ├── factory.py                 # 客户端工厂（按 provider 自动选型）
│   ├── context_manager.py         # 上下文管理（4种策略）
│   ├── tool_registry.py           # 工具注册中心（Function Calling）
│   └── clients/                   # 客户端实现
│       ├── __init__.py
│       ├── base.py                # 抽象基类（统一接口）
│       ├── openai_client.py       # OpenAI 格式（DeepSeek/Qwen/Kimi）
│       └── anthropic_client.py    # Anthropic 格式（Claude）
│
├── scripts/                       # 演示脚本（5个主题）
│   ├── 01_多环境多模型多轮对话.py
│   ├── 02_思考模式对比.py
│   ├── 03_工具调用对比.py
│   ├── 04_思考+工具+多轮交互.py
│   └── 05_上下文管理策略.py
│
├── .env.example                   # 环境变量模板
├── pyproject.toml                 # 项目配置（可 pip install -e .）
└── README.md                      # 本文档
```

---

## 快速开始

### 1. 安装依赖

```bash
cd tutorial-advanced-llm

# 基础安装（OpenAI 格式）
pip install -e .

# 完整安装（含 Anthropic）
pip install -e ".[all]"
```

### 2. 配置环境变量

```bash
# 复制模板
cp .env.example .env

# 编辑 .env，填入你的 API Key
```

### 3. 运行演示

```bash
# 运行单个脚本
python scripts/01_多环境多模型多轮对话.py
python scripts/02_思考模式对比.py
python scripts/03_工具调用对比.py
python scripts/04_思考+工具+多轮交互.py
python scripts/05_上下文管理策略.py

# 切换模型
$env:ACTIVE_MODEL="qwen"; python scripts/01_多环境多模型多轮对话.py

# 切换环境
$env:APP_ENV="prod"; python scripts/01_多环境多模型多轮对话.py
```

---

## 核心主题详解

### 主题一：多环境 + 多格式 + 多模型 + 多轮对话

**对应脚本**: `scripts/01_多环境多模型多轮对话.py`

#### 架构设计

```
┌─────────────────────────────────────────────────────────┐
│                    上层业务代码                           │
│         client.chat(messages) / chat_stream()           │
├─────────────────────────────────────────────────────────┤
│                  create_client() 工厂                    │
│         根据 provider 自动选择客户端实现                  │
├──────────────────────┬──────────────────────────────────┤
│   OpenAIClient       │      AnthropicClient             │
│   (DeepSeek/Qwen/    │      (Claude)                    │
│    Kimi/OpenAI)      │                                  │
├──────────────────────┴──────────────────────────────────┤
│              BaseLLMClient 抽象基类                       │
│         统一接口: chat() / chat_stream()                 │
├─────────────────────────────────────────────────────────┤
│              Config 配置管理                              │
│    .env → .env.models → .env.{APP_ENV} → 系统环境变量    │
└─────────────────────────────────────────────────────────┘
```

#### 多环境配置优先级

```
系统环境变量 > .env.{APP_ENV} > .env.models > .env
```

| 文件 | 用途 | 示例 |
|------|------|------|
| `.env` | 通用默认值 | `APP_ENV=dev`, `MAX_TOKENS=4096` |
| `.env.models` | 模型密钥（独立于环境） | `DEEPSEEK_API_KEY=sk-xxx` |
| `.env.dev` | 开发环境覆盖 | `DEBUG=true`, `LOG_LEVEL=DEBUG` |
| `.env.prod` | 生产环境覆盖 | `DEBUG=false`, `TIMEOUT=60` |

#### 多模型切换

```python
from advanced_llm import create_client, cfg

# 使用当前激活模型
client = create_client()

# 指定模型（工厂自动识别格式）
client = create_client("deepseek")   # → OpenAIClient
client = create_client("qwen")       # → OpenAIClient
client = create_client("kimi")       # → OpenAIClient
client = create_client("claude")     # → AnthropicClient

# 统一调用（无需关心底层格式）
result = client.chat([{"role": "user", "content": "你好"}])
print(result.content)
```

#### 多轮对话核心原理

```python
# API 是无状态的！每次请求必须带完整历史
messages = [{"role": "system", "content": "你是助手"}]

# 第1轮
messages.append({"role": "user", "content": "我叫小明"})
result = client.chat(messages)
messages.append({"role": "assistant", "content": result.content})  # 关键！

# 第2轮（带上了第1轮历史）
messages.append({"role": "user", "content": "我叫什么？"})
result = client.chat(messages)  # 模型能回答"你叫小明"
```

---

### 主题二：思考模式对比

**对应脚本**: `scripts/02_思考模式对比.py`

#### 各厂商思考控制参数

| 厂商 | 参数 | 控制粒度 | 示例 |
|------|------|---------|------|
| **DeepSeek** | `extra_body={"thinking": {"type": "enabled"}}` | 开/关 + 等级 | `reasoning_effort="high"` |
| **OpenAI** (o3/o4) | `reasoning_effort` | 粗粒度等级 | `"low"/"medium"/"high"` |
| **Anthropic** (Claude) | `thinking={"type":"enabled","budget_tokens":N}` | 精确 token 预算 | `budget_tokens=4096` |
| **Qwen** | 暂无原生参数 | — | 通过 prompt 引导 |
| **Kimi** | 暂无原生参数 | — | 通过 prompt 引导 |

#### 思考 vs 不思考的关键差异

| 维度 | 不思考 | 思考模式 |
|------|--------|---------|
| 响应结构 | 只有 `content` | `reasoning_content` + `content` |
| temperature | 可设置 | **不支持**（会报错） |
| 延迟 | 快 | 慢 2~5x |
| Token 消耗 | 少 | 多 3~10x |
| 适用场景 | 简单问答、翻译 | 数学推理、代码生成、复杂决策 |

#### 代码示例

```python
# 不思考
result = client.chat(messages, enable_thinking=False, temperature=0.7)

# 思考（DeepSeek）
result = client.chat(messages, enable_thinking=True)
print(result.reasoning)  # 思考过程
print(result.content)    # 最终回答
```

---

### 主题三：工具调用对比

**对应脚本**: `scripts/03_工具调用对比.py`

#### 使用工具 vs 不使用工具

| 维度 | 不使用工具 | 使用工具 |
|------|-----------|---------|
| finish_reason | `"stop"` | `"tool_calls"` |
| message.content | 回答文本 | `None`（还没回答完） |
| message.tool_calls | `None` | `[{id, function:{name, args}}]` |
| 后续操作 | 直接展示 | 执行工具 → 喂回结果 → 再调 API |

#### 完整工具调用流程

```
用户提问 → [模型判断需要工具] → 返回 tool_calls
    → 你的代码执行工具 → 拿到结果
    → 结果作为 role="tool" 喂回模型
    → 模型生成最终自然语言回答
```

#### 关键代码模式

```python
from advanced_llm import create_client, create_default_registry

client = create_client()
registry = create_default_registry()
tools = registry.get_openai_schemas()

# 第一次调用
result = client.chat(messages, tools=tools)

if result.has_tool_calls:
    # 把 assistant 的 tool_calls 加入历史
    messages.append({
        "role": "assistant",
        "content": result.content,
        "tool_calls": [{"id": tc.id, "type": "function",
                        "function": {"name": tc.name, "arguments": tc.arguments}}
                       for tc in result.tool_calls],
    })

    # 执行工具并喂回结果
    for tc in result.tool_calls:
        tool_result = registry.execute_tool_call(tc)
        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,  # 必须一一对应！
            "content": tool_result.result,
        })

    # 第二次调用：模型生成最终回答
    final = client.chat(messages, tools=tools)
    print(final.content)
```

#### tool_choice 控制

| 值 | 行为 | 适用场景 |
|----|------|---------|
| `"auto"` | 模型自行决定（默认） | 大多数场景 |
| `"none"` | 禁止使用工具 | 纯聊天、翻译 |
| `"required"` | 必须使用工具 | 必须查数据库 |
| `{"type":"function","function":{"name":"xxx"}}` | 强制指定 | 明确知道用哪个 |

---

### 主题四：思考 + 工具 + 多轮交互

**对应脚本**: `scripts/04_思考+工具+多轮交互.py`

#### 核心规则（DeepSeek 官方文档）

```
┌─────────────────────────────────────────────────────────────────┐
│  无工具调用时：reasoning_content 无需回传                        │
│  → 多轮历史中只保留 content（节省 token）                       │
│                                                                 │
│  有工具调用时：reasoning_content 必须回传                        │
│  → 后续所有轮次的 messages 中必须包含完整 reasoning              │
│  → 否则上下文断裂或 API 报错                                    │
└─────────────────────────────────────────────────────────────────┘
```

#### 代码对比

```python
# ═══ 场景A：无工具调用 → reasoning 可丢弃 ═══
result = client.chat(messages, enable_thinking=True)
messages.append({"role": "assistant", "content": result.content})
# reasoning 不传，节省 token ✅

# ═══ 场景B：有工具调用 → reasoning 必须保留 ═══
result = client.chat(messages, tools=tools, enable_thinking=True)
assistant_msg = {
    "role": "assistant",
    "content": result.content,
    "tool_calls": [...],
}
if result.reasoning:
    assistant_msg["reasoning_content"] = result.reasoning  # 必须！
messages.append(assistant_msg)
```

#### 生产级 Agent 循环

```python
max_iterations = 5
for i in range(max_iterations):
    result = client.chat(messages, tools=tools, enable_thinking=True)

    if not result.has_tool_calls:
        break  # 最终回答

    # 保留 reasoning + tool_calls
    messages.append(build_assistant_msg_with_reasoning(result))

    # 执行工具
    for tc in result.tool_calls:
        tool_result = registry.execute_tool_call(tc)
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": tool_result.result})
```

---

### 主题五：上下文管理策略

**对应脚本**: `scripts/05_上下文管理策略.py`

#### 四种策略对比

| 策略 | 原理 | 成本 | 信息保留 | 适用场景 |
|------|------|------|---------|---------|
| **滑动窗口** | 只保留最近 N 轮 | 低 | 低 | 闲聊、简单问答 |
| **Token 预算** | 按 token 数截断 | 中 | 中 | 精确成本控制 |
| **摘要压缩** | 用模型总结早期对话 | 高 | 高 | 长对话、客服 |
| **混合策略** | 窗口 + 预算 + 摘要 | 中 | 高 | **生产推荐** |

#### 使用方式

```python
from advanced_llm import ContextManager, HybridStrategy

# 创建管理器
cm = ContextManager(
    system_prompt="你是智能客服。",
    strategy=HybridStrategy(max_rounds=10, max_tokens=8000),
)

# 添加消息
cm.add_user_message("你好")
cm.add_assistant_message("你好！有什么可以帮你的？")

# 获取处理后的 messages（自动应用策略）
messages = cm.get_messages()

# 调用 API
result = client.chat(messages)
```

#### 成本增长对比

```
无管理（10轮）:  prompt_tokens 线性增长 → 5500+ tokens
滑动窗口（10轮）: prompt_tokens 稳定 → ~1000 tokens
节省: ~80%
```

---

## 支持的模型

| 模型 | Provider | 格式 | 思考模式 | 工具调用 |
|------|----------|------|---------|---------|
| DeepSeek V4 | openai | OpenAI | ✅ | ✅ |
| 通义千问 Qwen | openai | OpenAI | ❌ | ✅ |
| Kimi/Moonshot | openai | OpenAI | ❌ | ✅ |
| Claude | anthropic | Anthropic | ✅ | ✅ |

---

## 环境变量说明

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `APP_ENV` | `dev` | 环境标识（dev/prod） |
| `ACTIVE_MODEL` | `deepseek` | 当前激活模型 |
| `ENABLE_THINKING` | `false` | 是否开启思考模式 |
| `REASONING_EFFORT` | `high` | 思考深度（low/medium/high） |
| `MAX_TOKENS` | `4096` | 最大输出 token |
| `TEMPERATURE` | `0.7` | 温度（思考模式下无效） |
| `MAX_CONTEXT_TOKENS` | `8000` | 上下文最大 token |
| `MAX_HISTORY_ROUNDS` | `10` | 最大历史轮数 |
| `{MODEL}_API_KEY` | — | 各模型 API 密钥 |
| `{MODEL}_BASE_URL` | — | 各模型 API 地址 |
| `{MODEL}_MODEL` | — | 各模型标识符 |
| `{MODEL}_PROVIDER` | — | API 格式（openai/anthropic） |

---

## 设计模式

| 模式 | 应用位置 | 作用 |
|------|---------|------|
| **工厂模式** | `factory.py` | 根据 provider 自动选择客户端 |
| **模板方法** | `clients/base.py` | 统一接口，子类实现差异 |
| **策略模式** | `context_manager.py` | 可插拔的上下文管理策略 |
| **注册表模式** | `tool_registry.py` | 工具的统一注册与分发 |
| **单例模式** | `config.py` | 全局配置对象 `cfg` |
| **延迟导入** | `factory.py` | 按需加载 SDK，减少依赖 |

---

## 生产环境检查清单

- [ ] API Key 通过环境变量管理，不硬编码
- [ ] `.env` 文件加入 `.gitignore`
- [ ] 设置 `MAX_RETRIES` 和 `TIMEOUT`
- [ ] 工具调用设置 `max_iterations` 防死循环
- [ ] 上下文管理策略已配置（避免窗口溢出）
- [ ] 监控 token 消耗（按用户/按租户）
- [ ] 思考模式下 `max_tokens` 设置足够大（4096+）
- [ ] 工具执行有超时控制和错误处理
- [ ] 日志记录关键操作（工具调用、API 错误）
