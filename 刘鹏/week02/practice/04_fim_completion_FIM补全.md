# DeepSeek API 文档深度讲解：FIM 补全（FIM Completion）

> 原文出处：https://api-docs.deepseek.com/zh-cn/guides/fim_completion

---

## 1. 一句话概览

这个网页讲的是：**怎么让 AI 帮你"填空"——你给它一段代码的前半部分和后半部分，中间挖个洞，让 AI 把洞补上。**

打个生活比方：就像做填空题。普通对话是"请写一篇作文"（AI 从头写到尾）；而 FIM 是"从前有座山，____，庙里有个老和尚"——你给了开头和结尾，AI 只负责填中间那块。这在写代码时特别有用：你写到一半卡住了，或者想在一个函数中间插入一段逻辑，AI 能根据上下文把中间补全。

---

## 2. 全量专业术语扫盲

### 2.1 FIM 核心概念

- **FIM（Fill In the Middle，中间填充）**：直译就是"填中间"。这是一种特殊的代码补全模式：你提供代码的前文（prefix）和后文（suffix），AI 生成中间缺失的部分。这是专门为编程场景设计的。

- **prefix（前文）**：光标位置之前的代码。比如你写到第 10 行，前文就是第 1~10 行。

- **suffix（后文）**：光标位置之后的代码。比如你在第 10 行中间插入，后文就是第 10 行后半段 + 第 11 行到文件末尾。

- **completion（补全内容）**：AI 生成的、要填到中间的那段代码。

- **光标位置（Cursor Position）**：你希望 AI 补全的位置。前文和后文就是以光标为界切分的。

### 2.2 API 与端点相关

- **端点（Endpoint）**：API 的具体 URL 路径。不同的端点干不同的活。FIM 用的是 `/completions`，而普通对话用的是 `/chat/completions`。注意区别！

- **/completions（补全端点）**：旧式的、不带对话角色的补全接口。你给它一段文本，它接着往下写。FIM 走的就是这个端点。

- **/chat/completions（对话补全端点）**：带角色（system/user/assistant）的对话接口。前面几篇文档用的都是这个。

- **base_url**：DeepSeek 服务地址 `https://api.deepseek.com`。FIM 不需要 Beta 地址。

### 2.3 编程相关

- **model（模型名）**：FIM 补全要用专门的模型，如 `deepseek-coder`。这是 DeepSeek 专门为代码训练的模型，比通用模型更懂编程。

- **prompt**：在 `/completions` 端点里，输入不叫 messages，叫 prompt——就是一段纯文本，让模型接着写。

- **stop（停止序列）**：告诉 AI"遇到这几个字符就停"。FIM 场景下很重要，因为补全完中间代码后应该停下，不能继续往下瞎写后文。

---

## 3. 核心知识与参数全解

### 3.1 FIM 的工作原理

想象你在 VS Code 里写代码，光标停在某一行：

```python
def calculate_total(items):
    total = 0
    for item in items:
        # ← 光标在这里，你想让 AI 帮你补全循环体
    return total
```

FIM 的输入是：
- **prefix**（光标前）：`def calculate_total(items):\n    total = 0\n    for item in items:\n        `
- **suffix**（光标后）：`    return total`

AI 看到前后文，推断中间应该填什么，输出：
- **completion**：`total += item.price`

最终拼起来就是完整代码。**关键在于 AI 同时看到了"前面有什么"和"后面有什么"，所以补出来的东西能和两边都接上。**

### 3.2 FIM vs 普通对话补全的区别

| 维度 | 普通对话（chat/completions） | FIM（/completions） |
|------|---------------------------|---------------------|
| 端点 | `/chat/completions` | `/completions` |
| 输入格式 | messages 数组（带角色） | prompt + prefix + suffix |
| 是否看后文 | 否，只看前文（历史） | **是，前后都看** |
| 典型场景 | 聊天、问答、写文章 | 代码补全、填空 |
| 模型 | deepseek-chat 等 | deepseek-coder 等 |

### 3.3 所有相关参数详解

#### 3.3.1 请求参数

| 参数名 | 数据类型 | 默认值 | 作用说明 |
|--------|---------|--------|---------|
| `model` | 字符串 | 无（必填） | 模型名。FIM 用 `deepseek-coder`（专门为代码训练）。用错模型效果会差很多。 |
| `prompt` | 字符串 | 无（必填） | 输入文本。**FIM 场景下，这里放的是 prefix（前文）**。注意：虽然参数名叫 prompt，但 FIM 模式下它的角色是"前文"。 |
| `suffix` | 字符串 | 无（可选） | **后文**。这是 FIM 的核心参数——告诉 AI 光标后面还有什么。不传 suffix 就退化成普通补全（只看前文）。 |
| `max_tokens` | 整数 | 无（建议填） | 补全内容的最大长度。代码补全一般不用太长，256~1024 够用。 |
| `temperature` | 浮点数 | 1.0 | 随机性。代码补全建议低温度（0~0.3），保证输出稳定、可预测。 |
| `top_p` | 浮点数 | 1.0 | 另一种随机性控制。和 temperature 二选一。 |
| `stream` | 布尔值 | false | 是否流式输出。IDE 插件场景通常用流式，边生成边显示。 |
| `stop` | 字符串或数组 | 无 | 停止序列。AI 输出里一旦出现这些字符，立即停止。代码补全常用 `["\n\n", "\nclass ", "\ndef "]` 来在遇到下一个函数/类定义时停。 |
| `n` | 整数 | 1 | 一次生成几个候选补全。调试时可以设大点看多个选项，生产环境一般用 1。 |
| `echo` | 布尔值 | false | 是否在输出里回显 prompt。FIM 场景一般 false（只要补全部分）。 |
| `presence_penalty` | 浮点数 | 0 | 话题新颖度惩罚。代码场景一般不动。 |
| `frequency_penalty` | 浮点数 | 0 | 重复惩罚。代码场景一般不动。 |

#### 3.3.2 返回参数

| 字段名 | 数据类型 | 作用说明 |
|--------|---------|---------|
| `choices[0].text` | 字符串 | **补全内容**。注意是 `text` 不是 `message.content`（因为这是 completions 端点，不是 chat 端点）。 |
| `choices[0].finish_reason` | 字符串 | 停止原因。`stop` = 遇到停止序列或自然结束；`length` = 达到 max_tokens 被截断。 |
| `usage.prompt_tokens` | 整数 | 输入 token 数（prefix + suffix 都算）。 |
| `usage.completion_tokens` | 整数 | 补全的 token 数。 |
| `usage.total_tokens` | 整数 | 总和。 |

### 3.4 FIM 的拼接逻辑

最终代码 = prefix + completion + suffix

注意：API 返回的 `choices[0].text` 只有 completion 部分，prefix 和 suffix 都是你自己提供的，需要你自己拼。这点和前缀续写类似。

---

## 4. 代码示例剖析与环境配置指南

### 4.1 代码剖析

网页给出的 FIM 示例（整理后）：

```python
from openai import OpenAI

# 1. 创建客户端
client = OpenAI(
    api_key="<your api key>",
    base_url="https://api.deepseek.com",
)

# 2. 准备前文和后文
prefix = """
def quick_sort(arr):
"""
suffix = """
    return arr
"""

# 3. 调用 FIM 补全
response = client.completions.create(
    model="deepseek-coder",
    prompt=prefix,        # 前文
    suffix=suffix,        # 后文
    max_tokens=128,
)

# 4. 取出补全内容
print(response.choices[0].text)
```

**逐块解释：**

- **第 1 块（创建客户端）**：和之前一样，但注意 `base_url` 是普通地址 `https://api.deepseek.com`，不需要 `/beta`。

- **第 2 块（准备前后文）**：这是 FIM 的核心。`prefix` 是函数定义的开头 `def quick_sort(arr):`，`suffix` 是函数的结尾 `return arr`。**为什么这么切？** 因为我们想让 AI 补全函数体——排序逻辑。AI 看到"这是个快排函数"和"最后返回 arr"，就能推断中间应该写排序算法。

- **第 3 块（调用 completions.create）**：注意是 `client.completions.create`，不是 `client.chat.completions.create`。**为什么？** 因为 FIM 走的是 `/completions` 端点，不带对话角色。`prompt` 放前文，`suffix` 放后文，`model` 用 `deepseek-coder`（代码专用模型）。

- **第 4 块（取结果）**：`response.choices[0].text` 是补全内容（注意是 `.text` 不是 `.message.content`）。最终完整代码需要你自己拼：`prefix + completion + suffix`。

### 4.2 从零跑通指南

#### 第一步：安装依赖

```bash
pip install openai
```

#### 第二步：配置 API Key

参考前面文档，把密钥存到环境变量 `DEEPSEEK_API_KEY`。

#### 第三步：写代码

新建 `fim_test.py`：

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

# 模拟在 IDE 里写代码，光标停在函数体中间
prefix = """import json

def load_config(path):
    \"\"\"加载 JSON 配置文件\"\"\"
    # 光标停在这里，想补全函数体
"""

suffix = """    return config

if __name__ == "__main__":
    cfg = load_config("config.json")
    print(cfg)
"""

response = client.completions.create(
    model="deepseek-coder",
    prompt=prefix,
    suffix=suffix,
    max_tokens=256,
    temperature=0.2,  # 代码补全用低温度
    stop=["\n\n\n"],  # 遇到多个空行就停
)

completion = response.choices[0].text
full_code = prefix + completion + suffix

print("=== AI 补全的部分 ===")
print(completion)
print("\n=== 完整代码 ===")
print(full_code)
```

#### 第四步：运行

```bash
python fim_test.py
```

#### 第五步：预期输出

```
=== AI 补全的部分 ===
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)


=== 完整代码 ===
import json

def load_config(path):
    """加载 JSON 配置文件"""
    # 光标停在这里，想补全函数体
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config

if __name__ == "__main__":
    cfg = load_config("config.json")
    print(cfg)
```

看，AI 根据前文（要加载 JSON）和后文（return config），准确补出了"打开文件 + json.load"这段中间逻辑，而且和两边都接得上。

#### 常见错误排查

- **报错 "model not found"**：检查 model 是不是 `deepseek-coder`，别用成 `deepseek-chat`。
- **补全内容跑题**：检查 prefix 和 suffix 切分是否合理，前后文是否给了足够上下文。
- **补全太长停不下来**：调小 max_tokens，或加 stop 序列。
- **用了 chat.completions.create**：FIM 必须用 `client.completions.create`（注意没有 `chat`）。

---

## 5. 生产环境视角升华（纯技术向）

### 5.1 适用场景

- **IDE 代码补全插件**：用户在编辑器里敲代码，光标位置实时调 FIM，补出下一段。GitHub Copilot、Cursor 的核心能力就是这个。
- **代码填充/重构**：已有函数骨架（注释 + 签名 + return），让 AI 填函数体。
- **批量代码生成**：有模板代码，中间逻辑让 AI 补。比如 CRUD 接口，每个接口结构相似，中间业务逻辑不同。
- **测试用例生成**：函数实现是 prefix，测试框架的结尾是 suffix，AI 补中间的测试逻辑。
- **文档/注释补全**：函数签名是 prefix，函数体是 suffix，AI 补中间的 docstring。

### 5.2 避坑指南

- **前后文切分要精准**：FIM 效果高度依赖 prefix/suffix 的切分位置。切得不好（比如把一个完整的语法块从中间劈开），AI 会困惑，补出来的代码语法错误。理想切分点：语句结束、函数签名后、代码块开始处。

- **suffix 太短效果差**：如果 suffix 只有一两个字符，AI 几乎等于没看到后文，退化成普通补全。suffix 至少要给到当前函数/块的结尾。

- **延迟敏感**：IDE 补全要求低延迟（用户敲完键盘 200ms 内要出建议）。FIM 请求要尽量轻量：max_tokens 设小（64~128）、用流式、模型选小一点的。每次按键都触发请求会拖慢编辑器，要做防抖（debounce 300~500ms）。

- **补全质量不稳定**：同一前后文多次请求可能给出不同补全。生产环境建议 temperature=0（或接近 0），保证可复现。重要场景可以请求 n=3 取多个候选，按规则挑最好的。

- **token 成本**：prefix + suffix 都计费。长文件做 FIM 时，token 消耗不小。可以做"上下文裁剪"——只取光标前后 N 行作为 prefix/suffix，而不是整个文件。

- **安全：代码泄露风险**：FIM 会把用户代码片段发给 DeepSeek 服务器。涉及敏感代码（密钥、核心算法）的场景要谨慎，或做脱敏处理。

- **补全内容的语法验证**：AI 补的代码不一定能编译/运行。生产环境（如 IDE 插件）要做语法检查，补全结果语法不通过时不展示给用户。

### 5.3 最佳实践

**带防抖和缓存的 IDE 补全封装：**

```python
import os
import time
from openai import OpenAI
from functools import lru_cache

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

class CodeCompleter:
    def __init__(self, model="deepseek-coder", max_tokens=128, debounce_ms=400):
        self.model = model
        self.max_tokens = max_tokens
        self.debounce_s = debounce_ms / 1000
        self._last_call_time = 0
    
    def _trim_context(self, full_text, cursor_pos, max_lines_before=50, max_lines_after=30):
        """裁剪上下文，只取光标前后 N 行，控制 token 成本"""
        before = full_text[:cursor_pos].split("\n")
        after = full_text[cursor_pos:].split("\n")
        prefix = "\n".join(before[-max_lines_before:])
        suffix = "\n".join(after[:max_lines_after])
        return prefix, suffix
    
    def complete(self, full_text: str, cursor_pos: int):
        """根据光标位置做 FIM 补全"""
        # 防抖：距离上次调用太近就跳过
        now = time.time()
        if now - self._last_call_time < self.debounce_s:
            return None
        self._last_call_time = now
        
        prefix, suffix = self._trim_context(full_text, cursor_pos)
        
        try:
            response = client.completions.create(
                model=self.model,
                prompt=prefix,
                suffix=suffix,
                max_tokens=self.max_tokens,
                temperature=0.1,  # 代码补全低温度
                stop=["\n\n\n", "\nclass ", "\ndef ", "\nif __name__"],
            )
            completion = response.choices[0].text
            return completion
        except Exception as e:
            import logging
            logging.error(f"FIM completion failed: {e}")
            return None

# 使用
completer = CodeCompleter()
code = open("my_module.py").read()
suggestion = completer.complete(code, cursor_pos=500)
if suggestion:
    print("建议补全:", suggestion)
```

**多候选 + 评分选择：**

```python
def complete_with_candidates(prefix, suffix, n=3):
    """生成 n 个候选，按简单规则挑最好的"""
    response = client.completions.create(
        model="deepseek-coder",
        prompt=prefix,
        suffix=suffix,
        max_tokens=128,
        temperature=0.3,  # 稍高温度增加多样性
        n=n,
    )
    candidates = [c.text for c in response.choices]
    
    # 简单评分：优先选语法完整（以合理符号结尾）、长度适中的
    def score(text):
        s = 0
        if text.rstrip().endswith((";", "}", ")", ":")):
            s += 2  # 语句完整
        if 10 <= len(text) <= 200:
            s += 1  # 长度适中
        if "    " in text:  # 有缩进，像代码
            s += 1
        return s
    
    best = max(candidates, key=score)
    return best
```

**监控指标：**

- 补全采纳率（用户接受补全的比例）——核心产品指标
- 平均补全延迟 P50/P95/P99
- 补全触发到首 token 时间（流式场景）
- 单次补全 token 消耗分布
- 不同语言/文件类型的补全质量差异
