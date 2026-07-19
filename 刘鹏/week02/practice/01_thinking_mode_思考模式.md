# DeepSeek API 文档深度讲解：思考模式（Thinking Mode）

> 原文出处：https://api-docs.deepseek.com/zh-cn/guides/thinking_mode

---

## 1. 一句话概览

这个网页讲的是：**怎么让 DeepSeek 这个 AI 在回答你问题之前，先自己在"草稿纸"上偷偷想一会儿，把思路理清楚，然后再把最终答案告诉你。**

打个生活比方：你问一个数学题给一个学生，普通模式就像让他脱口而出答案；而"思考模式"就像让他先在草稿纸上列算式、推导一遍，再把工整的答案写在卷子上。前者快但容易出错，后者慢但更靠谱，尤其是面对复杂的推理题。

---

## 2. 全量专业术语扫盲

在正式拆解之前，先把网页里出现的"行话"全部翻译成大白话。你以后看任何 DeepSeek 相关资料，这些词都会反复出现，必须先建立直觉。

### 2.1 模型与 API 相关

- **DeepSeek-V3.2**：DeepSeek 公司发布的一个大语言模型版本号。你可以把它理解成"第三代第二版"的 AI 大脑。版本号越新，能力越强、bug 越少。思考模式就是从这个版本开始原生支持的。

- **API（Application Programming Interface，应用程序编程接口）**：你可以把它理解成"AI 餐厅的点餐窗口"。你（程序）不需要知道后厨怎么炒菜，只需要按菜单格式把需求递进窗口，AI 就会把结果递出来。DeepSeek API 就是 DeepSeek 公司对外开放的这个"点餐窗口"。

- **base_url（基础地址）**：相当于"餐厅地址"。你要先告诉程序去哪个网址找 DeepSeek 的服务。普通功能用 `https://api.deepseek.com`，Beta（测试版）功能要用 `https://api.deepseek.com/beta`。

- **api_key（API 密钥）**：相当于你的"会员卡号"。DeepSeek 要靠它识别你是谁、扣你多少钱。每个人注册后都能在官网生成自己专属的一串密钥，绝对不能泄露给别人。

### 2.2 思考模式专属术语

- **思考模式（Thinking Mode）**：模型在给出最终回答前，先输出一段"思考过程"（reasoning content），把推理步骤写出来，然后再给出正式答案。就像数学考试要求写解题过程。

- **reasoning_content（思考内容）**：模型"草稿纸"上的内容，也就是那段思考过程。它会单独放在一个字段里，和最终答案分开。

- **content（正文内容）**：模型最终给你的正式回答，相当于"卷子上写的答案"。

- **thinking_budget（思考预算）**：你给模型"思考"分配的 token 额度上限。token 可以粗略理解成"字"或"词"。设了预算，模型就不会无限制地想下去，防止它想太久浪费钱。

- **max_tokens（最大 token 数）**：整个回答（思考内容 + 正文内容）的总长度上限。超过这个数，模型就会被强制截断。

### 2.3 编程与调用相关

- **OpenAI SDK（OpenAI 软件开发工具包）**：OpenAI 公司提供的一个 Python 库（`openai`），用来方便地调用大模型 API。DeepSeek 的 API 接口设计跟 OpenAI 几乎一模一样，所以你可以直接用 OpenAI 的库来调 DeepSeek，只要把 `base_url` 和 `api_key` 换成 DeepSeek 的就行。这叫"兼容 OpenAI 接口"。

- **client（客户端对象）**：你用 `OpenAI(api_key=..., base_url=...)` 创建出来的一个对象，相当于"你和 AI 之间的专属联络员"。之后所有对话都通过它来发起。

- **chat.completions.create（对话补全创建）**：发起一次对话请求的方法名。`chat` 表示对话，`completions` 表示补全（让 AI 把话接下去），`create` 表示创建一次新的请求。

- **messages（消息列表）**：你传给模型的对话历史，是一个数组，里面每条消息都有 `role`（角色）和 `content`（内容）。角色分三种：`system`（系统设定，告诉 AI 它是谁）、`user`（用户说的话）、`assistant`（AI 之前说的话）。

- **stream（流式输出）**：相当于"边想边说"。普通模式是 AI 把整段话想完一次性给你；流式模式是 AI 一边生成一边往外吐，你能在屏幕上看到字一个一个蹦出来。体验更好，但代码处理稍复杂。

- **token**：大模型处理文本的最小单位。英文里大约 1 个 token = 0.75 个单词；中文里大约 1 个汉字 = 1~2 个 token。计费、长度限制都按 token 算。

### 2.4 其他

- **Beta（测试版）**：DeepSeek 把一些还在试验阶段、可能不稳定的功能放在 `/beta` 这个地址下。用 Beta 功能意味着你要承担"可能出 bug"的风险，但能提前尝鲜。

- **Sliding Window Attention（滑动窗口注意力）**：这是大模型底层的一种技术机制，简单说就是模型在处理超长文本时，只能"看到"附近一定范围内的内容，就像你读书时眼睛一次只能聚焦一段话。这个机制会影响缓存（见 kv_cache 文档），但思考模式本身不直接涉及。

---

## 3. 核心知识与参数全解

### 3.1 思考模式到底在干什么？

普通模式下，你问模型一个问题，它直接吐答案。思考模式下，流程变成两步：

1. **第一步：思考**。模型先生成一段 `reasoning_content`，里面是它的推理过程，比如"用户问的是 X，我需要先查 Y，然后算 Z……"。
2. **第二步：作答**。思考完之后，模型再生成正式的 `content`，也就是给用户看的答案。

这两段内容在 API 返回结果里是**分开存放**的：`reasoning_content` 在一个字段，`content` 在另一个字段。这样你就可以选择只把最终答案展示给用户，把思考过程藏起来（或者折叠显示）。

### 3.2 思考模式适合什么场景？

- **数学推理**：比如"鸡兔同笼"这种需要列方程的题。
- **逻辑分析**：比如"如果 A 则 B，如果 B 则 C，那么 A 成立时 C 成立吗"。
- **代码编写**：复杂的算法题，模型先想清楚思路再写代码，bug 更少。
- **复杂决策**：比如"帮我分析这三个方案的优劣并推荐一个"。

不适合的场景：简单的翻译、改写、闲聊——开了思考模式反而慢、还费钱。

### 3.3 所有相关参数逐一详解

下面把思考模式涉及的所有参数全部列出来，一个不漏。

#### 3.3.1 请求参数（你传给 API 的）

| 参数名 | 数据类型 | 默认值 | 作用说明 |
|--------|---------|--------|---------|
| `model` | 字符串 | 无（必填） | 指定用哪个模型。思考模式要用支持思考的模型，比如 `deepseek-v4-pro`（具体型号以官网最新为准）。填错了模型可能不支持思考。 |
| `messages` | 数组 | 无（必填） | 对话历史。每条消息是 `{"role": "system/user/assistant", "content": "..."}`。思考模式下，messages 的内容和普通模式一样写，区别只在模型内部处理方式。 |
| `max_tokens` | 整数 | 无（建议必填） | 整个回答（思考 + 正文）的最大 token 数。**注意：思考内容也算在内！** 如果你设了 1000，模型思考用了 800，那正文只剩 200 可用，可能答案没说完就被截断。所以开思考模式时 max_tokens 要设大一点。 |
| `stream` | 布尔值 | `false` | 是否流式输出。`true` 表示一边生成一边返回，`false` 表示全部生成完一次性返回。思考模式下流式输出体验更好，因为你能实时看到思考过程。 |
| `thinking_budget` | 整数 | 无 | **思考模式专属参数**。给模型的"思考"过程设定一个 token 预算上限。比如设 4096，模型最多思考 4096 个 token 就必须开始正式作答。这能防止模型在某些难题上无限纠结，控制成本。 |

#### 3.3.2 返回参数（API 返回给你的）

| 字段名 | 数据类型 | 出现条件 | 作用说明 |
|--------|---------|---------|---------|
| `choices[0].message.content` | 字符串 | 总是存在 | 模型的正式回答，也就是给用户看的最终答案。 |
| `choices[0].message.reasoning_content` | 字符串 | 仅思考模式下出现 | 模型的思考过程。**这是思考模式区别于普通模式的核心字段。** 你可以选择展示、折叠或丢弃它。 |
| `choices[0].finish_reason` | 字符串 | 总是存在 | 结束原因。`stop` 表示正常结束，`length` 表示被 max_tokens 截断了，`tool_calls` 表示模型要调用工具。 |
| `usage.prompt_tokens` | 整数 | 总是存在 | 你输入的 token 数。 |
| `usage.completion_tokens` | 整数 | 总是存在 | 模型输出的 token 数（**包含思考内容**）。 |
| `usage.total_tokens` | 整数 | 总是存在 | 输入 + 输出的总 token 数。 |

### 3.4 思考模式的关键注意事项

1. **思考内容也消耗 token、也花钱**：`reasoning_content` 不是免费的，它和 `content` 一样按 token 计费。所以思考模式比普通模式贵。
2. **max_tokens 要给够**：因为思考内容占额度，如果 max_tokens 设太小，可能思考没完就被截断，连正式答案都开始不了。
3. **思考内容可能为空**：如果问题太简单，模型可能"懒得想"，`reasoning_content` 为空字符串，直接给答案。这是正常的。
4. **思考模式与工具调用可叠加**：从 V3.2 开始，思考模式下也能用 tool_calls（详见 tool_calls 文档）。

---

## 4. 代码示例剖析与环境配置指南

### 4.1 代码剖析

网页给出的核心示例代码（整理后）如下：

```python
from openai import OpenAI

# 1. 创建客户端
client = OpenAI(
    api_key="<your api key>",
    base_url="https://api.deepseek.com",
)

# 2. 发起思考模式请求
response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=[
        {"role": "user", "content": "9.11 和 9.8 哪个大？"}
    ],
    # 思考模式相关参数
    # 注意：具体参数名以官网最新文档为准
)

# 3. 分别取出思考内容和正式答案
reasoning_content = response.choices[0].message.reasoning_content
content = response.choices[0].message.content

print("=== 思考过程 ===")
print(reasoning_content)
print("\n=== 最终答案 ===")
print(content)
```

**逐块解释：**

- **第 1 块（导入与创建客户端）**：`from openai import OpenAI` 是引入 OpenAI 的库。然后创建 `client`，把你的 API 密钥和 DeepSeek 的服务地址传进去。**为什么用 OpenAI 的库调 DeepSeek？** 因为 DeepSeek 故意把自己的接口做得和 OpenAI 一模一样，这样开发者换一家 AI 公司不用重写代码，只改地址和密钥就行。

- **第 2 块（发起请求）**：调用 `client.chat.completions.create(...)`，传入模型名、对话消息。`messages` 是一个列表，这里只放了一条 user 消息。模型会先思考"9.11 和 9.8 怎么比较"，再给出答案。

- **第 3 块（取结果）**：`response.choices[0].message` 是模型返回的消息对象。`.reasoning_content` 取思考过程，`.content` 取正式答案。**为什么要分开取？** 因为思考过程通常很长且含中间步骤，给终端用户看会显得啰嗦，所以你可以只展示 `content`，把 `reasoning_content` 记到日志里方便调试。

### 4.2 从零跑通指南（非科班友好）

#### 第一步：安装 Python

如果你电脑还没装 Python，去 [python.org](https://www.python.org/downloads/) 下载 3.10 以上版本，安装时**务必勾选 "Add Python to PATH"**。

打开终端（Windows 用 PowerShell，Mac 用 Terminal），输入：

```bash
python --version
```

能看到版本号就说明装好了。

#### 第二步：安装依赖库

在终端里执行：

```bash
pip install openai
```

这会安装 OpenAI 官方 Python 库。如果你有多个 Python 版本，可能需要用 `pip3` 或 `python -m pip install openai`。

#### 第三步：获取 DeepSeek API Key

1. 打开浏览器访问 https://platform.deepseek.com/
2. 注册账号并登录
3. 在左侧菜单找到"API Keys"或"访问令牌"
4. 点击"创建 API Key"，给它起个名字（比如"我的测试"）
5. **立刻复制生成的密钥并保存好**（页面关掉就再也看不到了，只能重新生成）

密钥长这样：`sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`

#### 第四步：配置 API Key（推荐用环境变量，别写死在代码里）

**为什么用环境变量？** 因为如果你把密钥直接写在代码里，一旦代码上传到 GitHub，全世界都能看到你的密钥，别人就能拿你的钱调 API。这是新手最容易踩的坑。

**Mac / Linux 操作：**

在终端执行（把 `sk-xxx` 换成你的真实密钥）：

```bash
echo 'export DEEPSEEK_API_KEY="sk-xxxxxxxxxxxxxxxx"' >> ~/.zshrc
source ~/.zshrc
```

**Windows 操作（PowerShell）：**

```powershell
setx DEEPSEEK_API_KEY "sk-xxxxxxxxxxxxxxxx"
```

设置完后**关闭并重新打开终端**，环境变量才生效。

验证是否设置成功：

```bash
# Mac/Linux
echo $DEEPSEEK_API_KEY
# Windows PowerShell
echo $env:DEEPSEEK_API_KEY
```

能看到你的密钥就说明 OK。

#### 第五步：写代码并运行

新建一个文件 `thinking_test.py`，内容如下：

```python
import os
from openai import OpenAI

# 从环境变量读取密钥（不要硬编码！）
client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

response = client.chat.completions.create(
    model="deepseek-v4-pro",  # 以官网最新支持的模型名为准
    messages=[
        {"role": "user", "content": "9.11 和 9.8 哪个大？请仔细思考。"}
    ],
)

msg = response.choices[0].message
print("=== 思考过程 ===")
print(getattr(msg, "reasoning_content", "(无思考内容)"))
print("\n=== 最终答案 ===")
print(msg.content)
```

在终端运行：

```bash
python thinking_test.py
```

#### 第六步：预期输出

输出大致长这样（具体内容模型每次可能不同）：

```
=== 思考过程 ===
用户问 9.11 和 9.8 哪个大。这看起来简单但容易出错。9.11 的小数部分是 0.11，9.8 的小数部分是 0.8。比较 0.11 和 0.8：0.11 = 11/100，0.8 = 80/100，显然 80 > 11，所以 0.8 > 0.11，因此 9.8 > 9.11。注意不要把 9.11 当成 9.11（版本号思维）或 9.8 当成 9.08。

=== 最终答案 ===
9.8 比 9.11 大。比较小数时，9.8 的小数部分 0.8 大于 9.11 的小数部分 0.11，所以 9.8 > 9.11。
```

如果看到 `AuthenticationError`，说明 API Key 没配对；如果看到 `ModelNotFound`，说明模型名写错了，去官网查最新模型名。

---

## 5. 生产环境视角升华（纯技术向）

### 5.1 适用场景

- **复杂推理类业务**：如数学辅导、法律案例分析、医疗诊断辅助、金融研报生成。这类任务对答案准确率要求高，思考模式带来的推理增益远超其额外成本。
- **Agent（智能体）决策链路**：在多步 Agent 系统中，让模型在每一步决策前先思考，能显著降低错误决策率。配合 tool_calls 使用，模型能先想清楚"该调哪个工具、传什么参数"再行动。
- **代码生成与审查**：复杂算法实现、bug 定位。模型先理清逻辑再写代码，可读性和正确率都更高。
- **可解释性要求高的场景**：如合规审计、教育辅导，需要把推理过程展示给用户或审计人员看，`reasoning_content` 天然满足这个需求。

### 5.2 避坑指南

- **max_tokens 截断陷阱**：思考内容计入 max_tokens，若预算不足，模型可能思考到一半被截断，`content` 为空或残缺。生产环境建议 max_tokens 至少设 4096 以上，并用 `finish_reason` 字段判断是否被截断（值为 `length` 表示被截断），触发重试或降级逻辑。

- **成本失控风险**：思考模式下单次请求的 token 消耗可能是普通模式的 3~10 倍。务必配合 `thinking_budget` 限制思考长度，并在业务层做 token 用量监控与告警。建议接入按用户/按租户的配额系统。

- **延迟翻倍**：思考过程本身需要生成时间，首 token 延迟（TTFT）和总延迟都会显著增加。对延迟敏感的实时对话场景，要评估是否值得开思考模式，或做"简单问题走普通模式、复杂问题走思考模式"的路由分流。

- **reasoning_content 的稳定性**：不同模型版本、不同请求下，`reasoning_content` 可能为空、为空格、或格式不统一。业务代码要做空值兜底，不要假设它一定有内容。

- **流式输出的 chunk 拼接**：流式模式下，`reasoning_content` 和 `content` 会分多个 chunk 返回，且可能交替出现。需要正确维护一个状态机，根据 chunk 里的 `delta` 字段判断当前是思考阶段还是作答阶段，分别累加到不同缓冲区。

- **不要把 reasoning_content 当作可信依据**：思考过程是模型"自言自语"，中间可能包含错误尝试、自我纠正，甚至幻觉。只能作为辅助参考，不能作为最终决策依据。

### 5.3 最佳实践

**架构层面：**

- **思考/非思考路由**：在网关层根据问题复杂度（可用一个轻量分类模型或规则判断）路由到不同模式，平衡成本与质量。
- **思考内容异步存储**：`reasoning_content` 通常很长，不要和主答案存在一张表，建议单独存到对象存储或日志系统，主表只存 `content` 和指向思考内容的引用 ID。
- **缓存思考结果**：对于相同或相似的问题，可缓存上次的思考过程与答案，避免重复消耗。配合 kv_cache 能进一步降低成本。

**代码层面：**

```python
import os
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def chat_with_thinking(user_msg: str, thinking_budget: int = 4096, max_tokens: int = 8192):
    """带重试、参数兜底的思考模式调用"""
    response = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=[{"role": "user", "content": user_msg}],
        max_tokens=max_tokens,
        # thinking_budget=thinking_budget,  # 以官网最新参数为准
    )
    choice = response.choices[0]
    
    # 检查是否被截断
    if choice.finish_reason == "length":
        # 记录日志，触发降级或重试
        import logging
        logging.warning(f"Response truncated for query: {user_msg[:50]}...")
    
    msg = choice.message
    return {
        "answer": msg.content,
        "reasoning": getattr(msg, "reasoning_content", "") or "",
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        },
        "finish_reason": choice.finish_reason,
    }
```

**监控指标建议：**

- 思考模式调用占比、平均思考 token 数、平均作答 token 数
- 因 `length` 截断的请求比例
- 思考模式 vs 普通模式的平均延迟对比
- 单用户/单租户的 token 消耗趋势，设置异常告警阈值
