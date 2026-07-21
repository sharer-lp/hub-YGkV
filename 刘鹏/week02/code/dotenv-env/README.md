# dotenv-env：多环境 + 多模型 + 多格式 LLM 客户端包

> 一个生产级的 Python 包，演示如何用**工厂模式 + 抽象基类**统一管理多种大模型 API。

## 你将学到什么

| 知识点 | 对应文件 | 说明 |
|--------|----------|------|
| Python 包结构 | `dotenv_env/__init__.py` | 如何组织一个可导入的包 |
| 抽象基类 (ABC) | `clients/base.py` | 定义统一接口，子类必须实现 |
| 工厂模式 | `factory.py` | 根据配置自动创建对应客户端 |
| 分层配置 | `config.py` + `.env*` | 多环境配置隔离与优先级 |
| dataclass | `models.py` | 轻量数据结构定义 |
| 延迟导入 | `factory.py` | 按需加载 SDK，减少启动开销 |

---

## 项目结构

```
dotenv-env/                          ← 项目根目录
│
├── dotenv_env/                      ← 🐍 Python 包（核心代码）
│   ├── __init__.py                  ← 包的"门面"：统一对外导出
│   ├── config.py                    ← 配置层：加载 .env 文件，暴露 cfg 单例
│   ├── models.py                    ← 数据层：ChatResult, StreamChunk
│   ├── factory.py                   ← 工厂层：create_client() 按 provider 创建客户端
│   └── clients/                     ← 实现层：各格式客户端（子包）
│       ├── __init__.py              ← 导出所有客户端类
│       ├── base.py                  ← 抽象基类（定义统一接口契约）
│       ├── openai_client.py         ← OpenAI 格式（DeepSeek / Qwen）
│       └── anthropic_client.py      ← Anthropic 格式（Claude）
│
├── main.py                          ← 演示入口（学习从这里开始）
├── pyproject.toml                   ← 包元数据（pip install -e . 用）
│
├── .env                             ← 通用默认配置（最先加载，决定环境）
├── .env.dev                         ← 开发环境覆盖
├── .env.prod                        ← 生产环境覆盖
├── .env.models                      ← 模型密钥（不提交 Git）
├── .env.example                     ← 配置模板（提交 Git）
│
└── README.md                        ← 本文件
```

### 设计思路：为什么这样分层？

#### 核心原则：每一层只做一件事

```
┌─────────────────────────────────────────────────────────────┐
│  __init__.py（门面层）                                       │
│  职责：决定"外部能看到什么"                                    │
│  规则：外部只需 from dotenv_env import create_client, cfg     │
│        内部怎么拆分、怎么实现，外部完全不用关心                   │
└─────────────────────────────────────────────────────────────┘
        │ 导出
        ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐
│  config.py   │  │  models.py   │  │  factory.py          │
│  配置层       │  │  数据层       │  │  工厂层              │
│              │  │              │  │                      │
│  只管读配置   │  │  只管定义结构 │  │  只管"选谁、创建谁"  │
│  不管业务逻辑 │  │  不含任何逻辑 │  │  不管具体怎么调用API  │
└──────────────┘  └──────────────┘  └──────────────────────┘
                                            │ 创建
                                            ▼
                                   ┌──────────────────────┐
                                   │  clients/（实现层）    │
                                   │                      │
                                   │  base.py  定接口契约  │
                                   │  openai   实现OpenAI  │
                                   │  anthropic 实现Claude │
                                   └──────────────────────┘
```

#### 为什么 models.py 要单独拆出来？

`ChatResult` 和 `StreamChunk` 被 **多个模块同时依赖**：
- `base.py` 的返回类型需要它
- `openai_client.py` 构造响应需要它
- `anthropic_client.py` 构造响应需要它
- 外部调用方也需要它（`result.content`）

如果放在 `base.py` 里，就形成了"所有人都依赖 base"的耦合。
拆出来后，`models.py` 是最底层的"零依赖"模块，谁都可以安全导入。

#### 为什么 clients/ 要做成子包？

- **隔离性**：每种 API 格式的差异很大（消息结构、流式事件、认证方式），放一起会互相干扰
- **可扩展**：新增 Gemini？加一个 `gemini_client.py` 就行，不动其他文件
- **延迟导入**：工厂只 import 你实际用的那个客户端，没装 anthropic 不影响 DeepSeek

#### 对比重构前（平铺）的问题

```
dotenv-env/
├── config.py          ← 6 个 .py 全挤一层，看不出谁依赖谁
├── base_client.py
├── openai_client.py
├── anthropic_client.py
├── client_factory.py
├── deepseek_client.py
└── main.py
```

| 问题 | 说明 |
|------|------|
| 不可导入 | 没有 `__init__.py`，外部无法 `from dotenv_env import ...` |
| 职责不清 | 打开目录看到 7 个平级文件，不知道从哪看起 |
| 耦合严重 | 数据类和基类混在一起，改一个怕影响另一个 |
| 扩展困难 | 再加 2 个 provider 就是 9 个文件平铺，越来越乱 |

---

## 快速开始

### 1. 安装依赖

```bash
pip install openai anthropic python-dotenv
```

### 2. 配置密钥

```bash
# 复制模板
cp .env.example .env
cp .env.example .env.models

# 编辑 .env.models，填入你的 API Key
```

### 3. 运行

```bash
python main.py
```

---

## 核心用法

### 最简调用（3 行代码）

```python
from dotenv_env import create_client

client = create_client()  # 自动读取配置，创建对应客户端
result = client.chat([{"role": "user", "content": "你好"}])
print(result.content)
```

### 切换模型

```python
from dotenv_env import create_client

# 方式1：代码中指定
client = create_client("deepseek")   # → OpenAIClient
client = create_client("qwen")       # → OpenAIClient
client = create_client("claude")     # → AnthropicClient

# 方式2：环境变量（不改代码）
# $env:ACTIVE_MODEL="qwen"; python main.py
```

### 流式输出

```python
from dotenv_env import create_client

client = create_client()
messages = [{"role": "user", "content": "讲个笑话"}]

# 方式1：自动打印
result = client.chat_stream_print(messages)

# 方式2：手动控制
for chunk in client.chat_stream(messages):
    if chunk.type == "reasoning":
        print(f"[思考] {chunk.data}", end="")
    elif chunk.type == "content":
        print(chunk.data, end="")
```

### 多轮对话

```python
from dotenv_env import create_client

client = create_client()
messages = [{"role": "user", "content": "我叫小明"}]

result = client.chat(messages)
print(result.content)

# 关键：用 build_assistant_message 拼接上下文（自动剥离 reasoning）
messages.append(client.build_assistant_message(result))
messages.append({"role": "user", "content": "我叫什么？"})

result2 = client.chat(messages)
print(result2.content)  # "你叫小明"
```

---

## 架构设计（重点学习）

### 整体调用链

```
你的代码
    │
    │  from dotenv_env import create_client
    ▼
┌─────────────────────────────────────────────────────┐
│  dotenv_env/__init__.py    （包的入口，统一导出）      │
└─────────────────────────────────────────────────────┘
    │
    │  create_client("deepseek")
    ▼
┌─────────────────────────────────────────────────────┐
│  factory.py                （工厂：根据 provider 选择）│
│                                                     │
│  provider="openai"    → OpenAIClient                │
│  provider="anthropic" → AnthropicClient             │
└─────────────────────────────────────────────────────┘
    │                              │
    ▼                              ▼
┌──────────────────┐    ┌──────────────────────┐
│ openai_client.py │    │ anthropic_client.py  │
│ (DeepSeek/Qwen)  │    │ (Claude)             │
└──────────────────┘    └──────────────────────┘
    │                              │
    └──────────┬───────────────────┘
               │ 继承
               ▼
    ┌─────────────────────┐
    │ clients/base.py     │
    │ (抽象基类)           │
    │ - chat()            │
    │ - chat_stream()     │
    └─────────────────────┘
               │
               │ 使用
               ▼
    ┌─────────────────────┐
    │ models.py           │
    │ - ChatResult        │
    │ - StreamChunk       │
    └─────────────────────┘
```

### 设计模式解析

#### 1. 抽象基类（Template Method）

```python
# clients/base.py
class BaseLLMClient(ABC):
    @abstractmethod
    def chat(self, messages) -> ChatResult: ...      # 子类必须实现

    @abstractmethod
    def chat_stream(self, messages) -> Generator: ... # 子类必须实现

    def chat_stream_print(self, messages) -> ChatResult:
        # 通用实现：调用 chat_stream() 并打印
        # 子类无需重复写打印逻辑
        ...
```

**好处**：上层代码只依赖 `BaseLLMClient` 接口，不关心底层是 OpenAI 还是 Anthropic。

#### 2. 简单工厂（Simple Factory）

```python
# factory.py
def create_client(model_name=None):
    model_cfg = cfg.get_model(model_name)       # 1. 读配置
    client_class = _get_client_class(provider)  # 2. 根据 provider 选类
    return client_class(model_name)             # 3. 实例化
```

**好处**：调用方不需要知道 `OpenAIClient` / `AnthropicClient` 的存在。

#### 3. 延迟导入（Lazy Import）

```python
# factory.py
_PROVIDER_IMPORTS = {
    "openai": ("dotenv_env.clients.openai_client", "OpenAIClient"),
    "anthropic": ("dotenv_env.clients.anthropic_client", "AnthropicClient"),
}

# 只在实际使用时才 import 对应模块
module = importlib.import_module(module_path)
```

**好处**：没装 `anthropic` SDK 也能正常使用 DeepSeek/Qwen。

---

## 配置系统

### 加载顺序 vs 覆盖优先级（两个不同的概念）

这是最容易混淆的点，我们分开讲：

#### 加载顺序（时间维度：谁先被读取）

```
第 1 步：加载 .env            ← 最先读取，目的是探测 APP_ENV=dev/prod
第 2 步：加载 .env.models     ← 读取模型密钥
第 3 步：加载 .env.{APP_ENV}  ← 根据第 1 步得到的环境名，加载对应文件
```

> **为什么 .env 必须第一个加载？**
> 因为 `.env` 里有 `APP_ENV=dev`，程序需要先知道"现在是哪个环境"，
> 才能决定第 3 步该加载 `.env.dev` 还是 `.env.prod`。
> 它是"引导文件"——决定后续流程的走向。

#### 覆盖优先级（效力维度：同名变量谁说了算）

```
系统环境变量        ← 最高（代码里 override=False 保证永远不被文件覆盖）
    ↑
.env.{APP_ENV}     ← 次高（override=True，可覆盖 .env 和 .env.models 的值）
    ↑
.env.models        ← 中等（override=False，不覆盖已有值）
    ↑
.env               ← 最低（最先加载，后面的文件都能覆盖它）
```

> **为什么 .env 优先级最低？**
> 它存的是"通用默认值"（如 `TIMEOUT=120`），是兜底用的。
> 开发环境想改成 180？写在 `.env.dev` 里覆盖就行，不用动 `.env`。
> 生产环境想改成 60？写在 `.env.prod` 里覆盖。
> 这样 `.env` 保持稳定，各环境按需覆盖，互不干扰。

#### 对应代码（config.py）

```python
def _load_env_files(self):
    # 第 1 步：加载 .env（探测环境）
    load_dotenv(".env", override=False)       # 不覆盖系统变量
    app_env = os.getenv("APP_ENV", "dev")     # 读出当前环境

    # 第 2 步：加载模型密钥
    load_dotenv(".env.models", override=False) # 不覆盖已有值

    # 第 3 步：加载环境专属配置
    load_dotenv(f".env.{app_env}", override=True)  # 覆盖 .env 的默认值
```

#### 一个具体例子

假设三个文件都定义了 `TIMEOUT`：

| 文件 | 值 | 角色 |
|------|-----|------|
| `.env` | `TIMEOUT=120` | 通用默认 |
| `.env.dev` | `TIMEOUT=180` | 开发环境宽松 |
| `.env.prod` | `TIMEOUT=60` | 生产环境严格 |

最终结果：
- `APP_ENV=dev` → 实际 TIMEOUT = **180**（.env.dev 覆盖了 .env）
- `APP_ENV=prod` → 实际 TIMEOUT = **60**（.env.prod 覆盖了 .env）
- 系统设了 `$env:TIMEOUT="30"` → 实际 TIMEOUT = **30**（系统变量最高）

### 环境切换

| 方式 | 命令 | 场景 |
|------|------|------|
| PowerShell 临时 | `$env:APP_ENV="prod"; python main.py` | 本地测试生产配置 |
| Linux/macOS | `APP_ENV=prod python main.py` | 同上 |
| 系统环境变量 | 永久配置 | 服务器部署 |
| 改 .env 文件 | 修改 `APP_ENV=prod` | 不推荐（失去隔离意义） |

### 配置变量一览

**通用配置（.env）：**

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `APP_ENV` | `dev` | 环境：`dev` / `prod` |
| `ENABLE_THINKING` | `false` | 思考模式开关 |
| `REASONING_EFFORT` | `high` | 思考强度 |
| `MAX_TOKENS` | `4096` | 最大输出 token |
| `TIMEOUT` | `120` | 超时（秒） |
| `MAX_RETRIES` | `3` | 重试次数 |
| `DEBUG` | `false` | 调试模式 |
| `LOG_LEVEL` | `INFO` | 日志级别 |

**模型配置（.env.models）：**

| 变量 | 说明 |
|------|------|
| `{ALIAS}_API_KEY` | API 密钥（必填） |
| `{ALIAS}_BASE_URL` | API 地址 |
| `{ALIAS}_MODEL` | 模型名称 |
| `{ALIAS}_PROVIDER` | 格式：`openai` / `anthropic` |
| `ACTIVE_MODEL` | 当前激活模型 |

---

## 思考模式

| 格式 | 控制方式 | 示例 |
|------|----------|------|
| OpenAI (DeepSeek V4) | `extra_body={"thinking": {"type": "enabled"}}` | 开/关二选一 |
| Anthropic (Claude) | `thinking={"type": "enabled", "budget_tokens": N}` | 可控制思考深度 |

**通用规则：**
- 思考模式下不支持 `temperature` / `top_p`
- 多轮对话必须剥离 reasoning（用 `build_assistant_message()`）

---

## 扩展新模型（3 步）

以添加 Gemini 为例：

**Step 1**：`config.py` 注册别名
```python
KNOWN_MODELS = ["deepseek", "qwen", "claude", "gemini"]
DEFAULT_PROVIDERS = { ..., "gemini": "openai" }  # Gemini 兼容 OpenAI 格式
```

**Step 2**：`.env.models` 添加配置
```bash
GEMINI_API_KEY=your-key
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
GEMINI_MODEL=gemini-2.0-flash
GEMINI_PROVIDER=openai
```

**Step 3**：使用
```python
client = create_client("gemini")  # 自动走 OpenAI 格式
```

如果是全新 API 格式（非 OpenAI/Anthropic），只需在 `clients/` 下新增一个文件继承 `BaseLLMClient`，然后在 `factory.py` 的 `_PROVIDER_IMPORTS` 中注册即可。

---

## 注意事项

1. **安全**：`.env` / `.env.models` 等含密钥文件不要提交 Git
2. **协作**：只提交 `.env.example`，团队成员复制后填自己的密钥
3. **代理**：若设了 `HTTP_PROXY`/`HTTPS_PROXY`，SDK 会自动读取导致连接失败
4. **安装**：开发模式安装 `pip install -e .`，即可在任意目录 `from dotenv_env import ...`
