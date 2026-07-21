# dotenv_env 包 · 深度教学手册

> 目标：让你不只是"会调用"，而是真正理解**为什么要这样写**，以后自己开发时脑子里有框架。

---

## 第一章：设计思想（最重要，先读这章）

### 1.1 这个项目到底在解决什么问题？

想象一个场景：你的项目需要调用大模型 API。

- DeepSeek 用 OpenAI 格式的 SDK
- Claude 用 Anthropic 格式的 SDK
- 以后可能还有 Gemini、文心一言……

**最笨的写法：**

```python
# 每次换模型，业务代码都要改
if 用deepseek:
    from openai import OpenAI
    client = OpenAI(...)
    resp = client.chat.completions.create(...)
elif 用claude:
    import anthropic
    client = anthropic.Anthropic(...)
    resp = client.messages.create(...)
```

问题：
- 业务代码里到处都是 `if/else`
- 换模型要改 N 个文件
- 新加一个模型，所有地方都要加 `elif`

**核心设计目标：让业务代码永远只写一种调用方式，不管底层是哪个模型。**

```python
# 理想状态：业务代码永远长这样，不管底层换了什么模型
client = create_client()
result = client.chat(messages)
print(result.content)
```

---

### 1.2 三大设计模式（本项目用到的）

#### 模式一：抽象基类（定义"契约"）

**生活类比：** 插座标准。

国标插座规定了"三个孔、220V、50Hz"。不管你插的是海尔空调还是格力空调，只要符合这个标准就能用。

代码里：
```python
class BaseLLMClient(ABC):
    def chat(self, messages) -> ChatResult: ...      # 契约：你必须能非流式调用
    def chat_stream(self, messages) -> Generator: ... # 契约：你必须能流式调用
```

所有客户端（OpenAI、Anthropic）都必须实现这两个方法。
上层代码只认识 `BaseLLMClient`，不关心你具体是谁。

**培养意识：** 当你发现"多个东西做类似的事，但细节不同"时，就该想到抽象基类。

---

#### 模式二：工厂模式（"你不用管怎么造，告诉我要什么就行"）

**生活类比：** 餐厅点菜。

你跟服务员说"来一份宫保鸡丁"，你不需要知道厨师是谁、用什么锅、几号灶。厨房（工厂）帮你搞定。

代码里：
```python
def create_client(model_name=None):
    # 1. 查配置：这个模型的 provider 是什么？
    # 2. 根据 provider 找到对应的客户端类
    # 3. 创建实例返回给你
```

调用方只需要 `create_client("claude")`，完全不用知道 `AnthropicClient` 这个类的存在。

**培养意识：** 当"创建对象的过程很复杂"或"创建哪种对象取决于配置"时，就该想到工厂。

---

#### 模式三：单例模式（"全局只需要一个"）

**生活类比：** 一个公司只需要一个 CEO。

配置信息全局只需要一份。你不想在每个文件里都重新读一遍 `.env` 文件。

代码里：
```python
# config.py 最后一行
cfg = Config()  # 创建一次，全局共享

# 任何地方都用同一个 cfg
from dotenv_env.config import cfg
print(cfg.TIMEOUT)  # 所有模块读到的是同一份配置
```

**培养意识：** 当某个对象"全局只需要一个实例"且"到处都要用"时，就该想到单例。

---

### 1.3 分层思想（最核心的架构意识）

```
┌─────────────────────────────────────────────────────────┐
│  你的业务代码（main.py）                                  │
│  只认识：create_client() 和 cfg                          │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  门面层（__init__.py）                                    │
│  职责：决定"外面的人能看到什么"                             │
│  原则：内部怎么改，外面不受影响                             │
└─────────────────────────────────────────────────────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  config.py   │ │  models.py   │ │  factory.py  │
│  配置层       │ │  数据层       │ │  工厂层      │
│              │ │              │ │              │
│ 只管读配置    │ │ 只管定义结构  │ │ 只管创建对象  │
│ 不管业务     │ │ 不含逻辑     │ │ 不管怎么调用  │
└──────────────┘ └──────────────┘ └──────────────┘
                                        │
                                        ▼
                              ┌──────────────────┐
                              │  clients/（实现层）│
                              │                  │
                              │  base.py  定契约  │
                              │  openai   实现A   │
                              │  anthropic 实现B  │
                              └──────────────────┘
```

**核心原则：每一层只做一件事，层与层之间单向依赖。**

- `models.py` 谁都不依赖（最底层）
- `config.py` 只依赖标准库和 dotenv
- `base.py` 依赖 `models.py`
- `openai_client.py` 依赖 `base.py` + `config.py` + `models.py`
- `factory.py` 依赖 `config.py` + `base.py`
- `__init__.py` 把大家聚合起来对外暴露

**培养意识：** 写代码前先问自己——"这个功能应该放在哪一层？"

---

### 1.4 开发流程（以后自己做类似项目的步骤）

```
第 1 步：想清楚"有几种变化"
        → 本项目：API 格式会变（OpenAI / Anthropic / 未来更多）

第 2 步：把"不变的"抽成接口
        → 不变的是：都要 chat()、都要 chat_stream()、都返回 ChatResult

第 3 步：把"变化的"做成实现类
        → OpenAIClient 处理 OpenAI 协议的细节
        → AnthropicClient 处理 Anthropic 协议的细节

第 4 步：做一个工厂，根据配置选择实现
        → create_client() 读 provider 字段，自动选

第 5 步：把配置独立管理
        → config.py + .env 文件，改配置不改代码

第 6 步：用 __init__.py 包装成"一键导入"
        → from dotenv_env import create_client, cfg
```

---

## 第二章：文件逐个精讲

### 2.1 models.py — 数据结构的"地基"

```python
from dataclasses import dataclass
from typing import Optional
```

**知识点：`@dataclass` 是什么？**

普通写法（啰嗦）：
```python
class ChatResult:
    def __init__(self, content, reasoning=None, usage=None, finish_reason=None):
        self.content = content
        self.reasoning = reasoning
        self.usage = usage
        self.finish_reason = finish_reason
```

dataclass 写法（一行搞定）：
```python
@dataclass
class ChatResult:
    content: str                          # 必填
    reasoning: Optional[str] = None       # 可选，默认 None
    usage: Optional[dict] = None
    finish_reason: Optional[str] = None
```

`@dataclass` 自动帮你生成 `__init__`、`__repr__`、`__eq__` 等方法。
你只需要声明"有哪些字段"就行。

**知识点：`Optional[str]` 是什么？**

意思是"这个值可能是 str，也可能是 None"。
等价于 `str | None`（Python 3.10+ 写法）。

**为什么 models.py 要单独一个文件？**

因为 `ChatResult` 被很多地方用到：
- `base.py` 的返回类型要写它
- `openai_client.py` 创建它
- `anthropic_client.py` 创建它
- 外部调用方要读它的 `.content`

如果放在某个 client 里，其他 client 就要 import 它 → 互相依赖 → 乱套。
单独放一个文件 = "公共数据定义"，谁都可以安全导入，不会循环依赖。

---

### 2.2 config.py — 配置管理（最复杂的文件）

#### 整体结构

```
config.py
├── ConfigError        ← 自定义异常
├── ModelConfig        ← 一个模型的配置（dataclass）
├── Config             ← 配置主类（核心）
│   ├── __init__()     ← 加载配置
│   ├── _load_env_files()  ← 加载 .env 文件
│   ├── _load_models()     ← 加载所有模型
│   ├── active_model       ← 当前激活的模型
│   ├── get_model()        ← 获取指定模型
│   └── _get/_int/_bool    ← 读环境变量的工具方法
└── cfg = Config()     ← 全局单例
```

#### 关键代码逐行解读

**路径定位：**
```python
_BASE_DIR = Path(__file__).resolve().parent.parent
```
- `__file__` → 当前文件路径：`dotenv_env/config.py`
- `.resolve()` → 变成绝对路径
- `.parent` → 上一级：`dotenv_env/`
- `.parent.parent` → 再上一级：`dotenv-env/`（.env 文件在这里）

**环境文件加载（核心逻辑）：**
```python
def _load_env_files(self):
    # 第 1 步：加载 .env（探测当前是什么环境）
    load_dotenv(".env", override=False)
    app_env = os.getenv("APP_ENV", "dev")  # 读出 "dev" 或 "prod"

    # 第 2 步：加载模型密钥
    load_dotenv(".env.models", override=False)

    # 第 3 步：根据环境名加载对应文件
    load_dotenv(f".env.{app_env}", override=True)
```

**`override=False` vs `override=True` 的区别：**
- `False`：如果系统环境变量已经有这个值，就不覆盖（系统变量最大）
- `True`：不管有没有，直接覆盖（环境文件说了算）

**加载顺序 vs 优先级：**

| 概念 | 说明 |
|------|------|
| 加载顺序（时间） | `.env` 第一个加载 → 它是"引导文件"，告诉程序该用 dev 还是 prod |
| 覆盖优先级（效力） | 系统变量 > `.env.prod` > `.env.models` > `.env` |

`.env` 最先加载但优先级最低——因为它是"兜底默认值"，后面的文件可以覆盖它。

**自定义异常：**
```python
class ConfigError(Exception):
    pass
```

为什么不直接用 `Exception`？
- 精确捕获：`except ConfigError` 只抓配置错误，不会误抓其他异常
- 语义清晰：看到 `raise ConfigError` 就知道是配置出了问题

**`@property` 装饰器：**
```python
@property
def active_model(self) -> ModelConfig:
    return self._models[self._active_model_name]
```

效果：像访问属性一样调用方法。
```python
cfg.active_model          # 看起来像属性，实际执行了方法
cfg.active_model()        # ❌ 不用加括号
```

好处：外部不知道内部是字典还是列表，你随时可以改实现而不影响调用方。

**`@staticmethod` 静态方法：**
```python
@staticmethod
def _get(key: str, default: str = "") -> str:
    return os.getenv(key, default)
```

- 不需要访问 `self`（不依赖实例状态）
- 放在类里只是为了"归类"（这些都是配置读取的工具）
- 用 `self._get(...)` 或 `Config._get(...)` 都行

---

### 2.3 clients/base.py — 抽象基类（契约）

```python
from abc import ABC, abstractmethod
```

**`ABC` = Abstract Base Class（抽象基类）**

规则：
1. 继承 `ABC` 的类**不能直接实例化**
2. 标了 `@abstractmethod` 的方法，子类**必须实现**，否则报错

```python
class BaseLLMClient(ABC):

    @abstractmethod
    def chat(self, messages) -> ChatResult:
        ...  # 没有实现，只是"声明"

    @abstractmethod
    def chat_stream(self, messages) -> Generator:
        ...
```

如果你写：
```python
client = BaseLLMClient()  # ❌ TypeError: 不能实例化抽象类
```

如果你写了子类但没实现抽象方法：
```python
class MyClient(BaseLLMClient):
    pass

MyClient()  # ❌ TypeError: 没有实现 chat() 和 chat_stream()
```

**通用方法（子类不用重写）：**
```python
def chat_stream_print(self, messages) -> ChatResult:
    for chunk in self.chat_stream(messages):  # 调用子类的 chat_stream
        if chunk.type == "reasoning":
            print(chunk.data, end="")
        elif chunk.type == "content":
            print(chunk.data, end="")
    ...
```

这就是**模板方法模式**：
- 父类定义"流程骨架"（先打印思考、再打印回答、最后返回结果）
- 子类只填"具体步骤"（怎么获取 chunk）

**`Generator` 生成器：**
```python
def chat_stream(self, messages) -> Generator[StreamChunk, None, None]:
```

生成器 = 用 `yield` 代替 `return` 的函数：
```python
# 普通函数：一次性返回所有结果
def get_all():
    return [1, 2, 3, 4, 5]

# 生成器：一个一个产出，省内存
def get_one_by_one():
    yield 1
    yield 2
    yield 3
```

流式输出用生成器完美匹配：模型每产出一个词，就 `yield` 一个 chunk。

---

### 2.4 clients/openai_client.py — OpenAI 格式实现

**继承 + 实现抽象方法：**
```python
class OpenAIClient(BaseLLMClient):  # 继承基类

    def chat(self, messages) -> ChatResult:       # 实现抽象方法
        response = self.client.chat.completions.create(**kwargs)
        return ChatResult(content=..., reasoning=..., usage=...)

    def chat_stream(self, messages) -> Generator:  # 实现抽象方法
        for chunk in stream:
            yield StreamChunk(type="content", data=chunk.choices[0].delta.content)
```

**`**kwargs` 解包：**
```python
kwargs = {"model": "deepseek-v4-flash", "max_tokens": 4096, "stream": False}
self.client.chat.completions.create(**kwargs)
# 等价于：
self.client.chat.completions.create(model="deepseek-v4-flash", max_tokens=4096, stream=False)
```

`**kwargs` = 把字典展开成关键字参数。这样可以用一个字典动态构建参数。

**`getattr` 安全访问：**
```python
reasoning = getattr(message, "reasoning_content", None)
```

等价于：
```python
if hasattr(message, "reasoning_content"):
    reasoning = message.reasoning_content
else:
    reasoning = None
```

为什么需要？因为不是所有模型都返回 `reasoning_content` 字段。
直接 `message.reasoning_content` 会报 `AttributeError`。

**异常处理：**
```python
try:
    response = self.client.chat.completions.create(**kwargs)
except (APITimeoutError, RateLimitError, APIError) as e:
    logger.error(f"[{self.model_name}] 调用失败: {e}")
    raise  # 记录日志后继续抛出，让上层决定怎么处理
```

`raise` 不带参数 = 重新抛出当前异常。
目的：在这一层记个日志，但不吞掉异常（上层可能还要做重试等处理）。

---

### 2.5 clients/anthropic_client.py — Anthropic 格式实现

**与 OpenAI 的核心差异：**

| 差异点 | OpenAI 格式 | Anthropic 格式 |
|--------|-------------|----------------|
| system 消息 | 放在 messages 数组里 | 必须单独传 `system` 参数 |
| 思考模式 | `extra_body={"thinking": {"type": "enabled"}}` | `thinking={"budget_tokens": 10000}` |
| 响应结构 | `response.choices[0].message.content` | `response.content` (blocks 列表) |
| 流式事件 | `chunk.choices[0].delta.content` | `event.delta.text` / `event.delta.thinking` |

**`_split_system_message` 方法：**
```python
@staticmethod
def _split_system_message(messages):
    system_parts = []
    filtered = []
    for msg in messages:
        if msg["role"] == "system":
            system_parts.append(msg["content"])  # 挑出 system
        else:
            filtered.append(msg)                  # 保留其他
    return "\n\n".join(system_parts), filtered
```

为什么要这样做？
因为 Anthropic 的 API 不接受 messages 里有 system 角色。
但上层代码统一传 `[{"role": "system", ...}, {"role": "user", ...}]`。
所以这一层要"翻译"——把 system 剥离出来单独传。

**这就是抽象基类的价值：** 上层代码永远用统一的 messages 格式，
每个 client 内部自己处理"翻译"工作。

---

### 2.6 factory.py — 工厂（连接配置和客户端）

**核心逻辑只有 3 步：**
```python
def create_client(model_name=None):
    # 1. 查配置
    model_cfg = cfg.get_model(model_name) if model_name else cfg.active_model

    # 2. 根据 provider 找类
    client_class = _get_client_class(model_cfg.provider)

    # 3. 创建实例
    return client_class(model_name)
```

**延迟导入（Lazy Import）：**
```python
_PROVIDER_IMPORTS = {
    "openai": ("dotenv_env.clients.openai_client", "OpenAIClient"),
    "anthropic": ("dotenv_env.clients.anthropic_client", "AnthropicClient"),
}

# 不是一开始就 import，而是用到时才 import
module = importlib.import_module("dotenv_env.clients.anthropic_client")
client_class = getattr(module, "AnthropicClient")
```

为什么不在文件顶部直接 `from ... import AnthropicClient`？
因为 `anthropic_client.py` 的第一行就 `import anthropic`。
如果用户没装 anthropic 这个包，顶部导入会直接报错，连 DeepSeek 都用不了。

延迟导入 = 你用到谁才导入谁，没装 anthropic 不影响 DeepSeek。

**缓存机制：**
```python
_CLIENT_CACHE: dict[str, type] = {}

def _get_client_class(provider):
    if provider in _CLIENT_CACHE:     # 第二次直接返回，不重复 import
        return _CLIENT_CACHE[provider]
    ...
    _CLIENT_CACHE[provider] = client_class  # 第一次导入后缓存
    return client_class
```

---

### 2.7 `__init__.py` — 包的"门面"

```python
from dotenv_env.config import cfg, Config, ConfigError, ModelConfig
from dotenv_env.models import ChatResult, StreamChunk
from dotenv_env.clients.base import BaseLLMClient
from dotenv_env.factory import create_client, create_all_clients

__all__ = ["create_client", "cfg", "ChatResult", ...]
```

**`__init__.py` 的作用：**

没有它：
```python
from dotenv_env.factory import create_client  # 要写完整路径
from dotenv_env.config import cfg
```

有了它：
```python
from dotenv_env import create_client, cfg  # 一行搞定
```

**`__all__` 的作用：**

控制 `from dotenv_env import *` 时导出什么。
也相当于"文档"——告诉读者"这个包对外提供这些东西"。

---

## 第三章：Python 知识点速查表

| 语法 | 含义 | 本项目哪里用了 |
|------|------|----------------|
| `@dataclass` | 自动生成 `__init__` 等方法的装饰器 | models.py, config.py |
| `Optional[str]` | 值可以是 str 或 None | models.py 所有字段 |
| `ABC` + `@abstractmethod` | 抽象基类，强制子类实现方法 | base.py |
| `@property` | 把方法变成"属性"访问 | config.py 的 active_model |
| `@staticmethod` | 不需要 self 的方法 | config.py 的 _get/_int/_bool |
| `yield` | 生成器，逐个产出值 | chat_stream() |
| `**kwargs` | 字典解包为关键字参数 | 所有 API 调用 |
| `getattr(obj, "x", None)` | 安全获取属性，不存在返回 None | 提取 reasoning_content |
| `importlib.import_module()` | 运行时动态导入模块 | factory.py 延迟导入 |
| `Path(__file__).resolve()` | 获取当前文件的绝对路径 | config.py 定位 .env |
| `f"...{var}"` | f-string 格式化 | 到处都是 |
| `list[dict]` | 类型注解：字典组成的列表 | messages 参数 |
| `-> ChatResult` | 返回值类型注解 | 所有方法 |

---

## 第四章：调用链路图（一次完整调用发生了什么）

```python
# 你写的代码
from dotenv_env import create_client
client = create_client()
result = client.chat([{"role": "user", "content": "你好"}])
print(result.content)
```

**背后发生了什么：**

```
1. from dotenv_env import create_client
   → 执行 __init__.py
   → 触发 config.py 的 cfg = Config()
   → Config.__init__() 加载 .env → .env.models → .env.dev
   → 注册所有模型，确定 active_model = deepseek

2. create_client()
   → 读 cfg.active_model → provider = "openai"
   → _get_client_class("openai")
   → importlib 导入 openai_client.py → 拿到 OpenAIClient 类
   → OpenAIClient(None) → 创建 OpenAI SDK 实例

3. client.chat(messages)
   → _build_kwargs() 构建参数（model, max_tokens, thinking...）
   → self.client.chat.completions.create(**kwargs) 发 HTTP 请求
   → 解析响应 → 包装成 ChatResult 返回

4. result.content
   → 直接读 ChatResult 的 content 字段
```

---

## 第五章：设计意识总结（背下来）

| 场景 | 该想到什么 | 本项目对应 |
|------|-----------|-----------|
| 多个东西做类似的事 | 抽象基类 | BaseLLMClient |
| 创建哪种对象取决于配置 | 工厂模式 | create_client() |
| 全局只需要一份的东西 | 单例 | cfg = Config() |
| 改配置不想改代码 | 环境变量 + .env 文件 | config.py |
| 外部不该知道内部细节 | 封装 + 门面 | __init__.py |
| 一个文件被太多人依赖 | 拆到独立模块 | models.py |
| 不同 API 格式差异大 | 适配器（每个 client 内部"翻译"） | _split_system_message |
| 可选依赖不想强制安装 | 延迟导入 | importlib |

**记住一句话：好的代码 = 改一个地方不需要改其他十个地方。**

---

## 第六章：动手练习（巩固用）

### 练习 1：加一个新模型（不需要写代码）

假设要加 Gemini（兼容 OpenAI 格式），你需要：
1. `config.py` 的 `KNOWN_MODELS` 加 `"gemini"`
2. `DEFAULT_PROVIDERS` 加 `"gemini": "openai"`
3. `.env.models` 加 `GEMINI_API_KEY=xxx` 等变量
4. 调用：`create_client("gemini")`

思考：为什么不需要写新的 client 类？（因为 Gemini 兼容 OpenAI 格式）

### 练习 2：加一个全新格式（需要写代码）

假设要加一个"百度文心"格式（不兼容 OpenAI），你需要：
1. `clients/` 下新建 `wenxin_client.py`
2. 写 `class WenxinClient(BaseLLMClient)` 实现 `chat()` 和 `chat_stream()`
3. `factory.py` 的 `_PROVIDER_IMPORTS` 加 `"wenxin": ("dotenv_env.clients.wenxin_client", "WenxinClient")`
4. `config.py` 的 `DEFAULT_PROVIDERS` 加 `"wenxin": "wenxin"`

思考：为什么业务代码（main.py）完全不用改？

### 练习 3：读代码回答

1. 如果 `.env.models` 里没配 `QWEN_API_KEY`，`cfg.available_models` 会包含 "qwen" 吗？
2. `override=False` 和 `override=True` 的区别是什么？为什么 `.env.{APP_ENV}` 用 True？
3. 为什么 `base.py` 的 `__init__` 里用 `from dotenv_env.config import cfg` 而不是顶部导入？

<details>
<summary>答案（先自己想再看）</summary>

1. 不会。`_load_models()` 里有 `if api_key:` 判断，没配 key 的模型不注册。
2. False = 不覆盖已有值（保护系统环境变量）；True = 强制覆盖（环境文件要能覆盖 .env 的默认值）。
3. 避免循环导入。`config.py` 创建 `cfg` 时可能触发其他模块，如果 base.py 顶部就 import cfg，可能形成 A→B→A 的循环。放在方法内部 = 用到时才导入，此时一切已初始化完毕。

</details>
