# DeepSeek API 文档深度讲解：前缀续写（Chat Prefix Completion）

> 原文出处：https://api-docs.deepseek.com/zh-cn/guides/chat_prefix_completion

---

## 1. 一句话概览

这个网页讲的是：**怎么强行规定 AI 回复的开头几个字，让 AI 只能顺着你的开头把剩下的话接完。**

打个生活比方：你让朋友讲个故事，但要求他必须用"从前有座山"开头。朋友本来想从"很久很久以前"讲起，但被你强制了开头，只好顺着"从前有座山"往下编。这个功能在 AI 场景里就叫"前缀续写"——你给个开头，AI 负责把后半段补全。

---

## 2. 全量专业术语扫盲

### 2.1 核心概念

- **前缀续写（Prefix Completion）**：你提供一个文本开头（前缀），AI 不能自己重新起头，必须从你给的开头继续往下写。这是这个功能的核心。

- **Chat Prefix Completion（对话前缀补全）**：特指在对话（chat）场景下的前缀续写。也就是在 `messages` 里，最后一条 assistant 消息的内容，会被当作 AI 回复的开头，AI 要接着它写。

- **prefix（前缀）**：你指定的那段开头文本。在 API 里，它就是最后一条 `assistant` 角色消息的 `content`。

- **补全（Completion）**：AI 把前缀后面的内容生成出来。和"对话"的区别在于：对话是 AI 从零开始想怎么回就怎么回；补全是 AI 被你"开了个头"，只能往下接。

### 2.2 API 与参数相关

- **base_url（基础地址）**：DeepSeek 的服务地址。这个功能属于 Beta（测试版），所以要用 `https://api.deepseek.com/beta`，注意末尾多了 `/beta`。

- **Beta 功能**：还在测试阶段的功能，可能不稳定、可能未来改动接口。用 Beta 地址访问意味着你接受这些风险。

- **extra_body**：OpenAI Python 库提供的一个参数，用来传一些**非标准**的额外字段。因为 `enable_prefix` 不是 OpenAI 原生支持的参数，DeepSeek 自己加的，所以要通过 `extra_body` 传进去。

- **enable_prefix**：开关参数，设为 `true` 表示"启用前缀续写"，告诉 DeepSeek 把最后一条 assistant 消息当前缀用。

### 2.3 编程相关

- **assistant 消息**：messages 数组里 role 为 `assistant` 的消息，通常代表 AI 之前的回复。但在这里，它被"借用"来表示你希望 AI 这次回复的开头。

- **messages 数组**：对话历史。前缀续写场景下，最后一条必须是 assistant 消息，它的 content 就是前缀。

---

## 3. 核心知识与参数全解

### 3.1 前缀续写的运作机制

普通对话流程：
```
messages = [
    {role: user, content: "写一首关于春天的诗"}
]
→ AI 自由发挥："春风拂面花满枝，燕子归来柳丝垂……"
```

前缀续写流程：
```
messages = [
    {role: user, content: "写一首关于春天的诗"},
    {role: assistant, content: "春风"}   ← 这就是前缀！
]
enable_prefix = true
→ AI 必须从"春风"接着写："春风拂面花满枝，燕子归来柳丝垂……"
```

**关键点**：最后一条 assistant 消息的 content 不会被 AI 重复输出，AI 是从这段内容**之后**开始生成的。最终你拿到的回复，是前缀 + AI 续写的内容拼起来。

### 3.2 为什么需要这个功能？

1. **控制开头格式**：比如你要求 AI 回复必须以 `{"status": "success"` 开头（JSON 格式），用前缀续写能强制保证。
2. **引导风格/语气**：让 AI 用特定口吻起头，比如"亲爱的用户您好："，AI 就会顺着客服语气往下说。
3. **续写半成品**：你写了一半的报告，让 AI 接着写完。
4. **结构化输出**：配合 JSON 模式，强制 AI 输出符合特定 schema 的内容。

### 3.3 所有相关参数详解

#### 3.3.1 请求参数

| 参数名 | 数据类型 | 默认值 | 作用说明 |
|--------|---------|--------|---------|
| `model` | 字符串 | 无（必填） | 模型名。前缀续写需要用支持的模型（如 `deepseek-chat`、`deepseek-v4-pro` 等，以官网为准）。 |
| `messages` | 数组 | 无（必填） | 对话历史。**前缀续写时，最后一条必须是 assistant 消息**，其 content 就是前缀。 |
| `extra_body` | 字典 | 无 | OpenAI 库的扩展参数容器。里面放 `{"enable_prefix": true}` 来开启前缀续写。 |
| `extra_body.enable_prefix` | 布尔值 | false | **核心开关**。true = 启用前缀续写，最后一条 assistant 消息作为前缀；false或不传 = 普通对话模式。 |
| `max_tokens` | 整数 | 无（建议填） | 续写部分的最大长度（不包含前缀本身）。 |
| `stream` | 布尔值 | false | 是否流式输出。前缀续写常配合流式，实时看到 AI 接出来的内容。 |
| `temperature` | 浮点数 | 1.0 | 控制随机性。续写场景通常用较低温度（0.3~0.7）保证连贯。 |

#### 3.3.2 base_url 的特殊要求

| base_url 值 | 适用场景 |
|-------------|---------|
| `https://api.deepseek.com` | 普通功能 |
| `https://api.deepseek.com/beta` | **前缀续写必须用这个**（Beta 功能） |

**为什么需要 Beta 地址？** 因为前缀续写依赖 `enable_prefix` 这个非标准参数，DeepSeek 把它放在 Beta 通道里灰度发布。用普通地址传 `enable_prefix` 会被忽略，等于没开。

#### 3.3.3 messages 数组的特殊结构

前缀续写时，messages 必须这样结尾：

```python
messages = [
    {"role": "user", "content": "你的问题"},
    {"role": "assistant", "content": "你希望 AI 回复的开头"}  # ← 必须是最后一条
]
```

如果最后一条是 user 消息，即使开了 `enable_prefix`，也没有前缀可用，等于普通对话。

### 3.4 前缀续写 vs 普通对话的对比

| 维度 | 普通对话 | 前缀续写 |
|------|---------|---------|
| AI 回复开头 | AI 自由决定 | 你强制指定 |
| messages 最后一条 | 通常是 user | 必须是 assistant |
| 需要的 base_url | 普通 | Beta |
| 需要的开关 | 无 | `enable_prefix: true` |
| 适用场景 | 通用 | 格式控制、续写、引导 |

---

## 4. 代码示例剖析与环境配置指南

### 4.1 代码剖析

网页给出的示例（整理后）：

```python
from openai import OpenAI

# 1. 创建客户端，注意 base_url 是 beta 地址
client = OpenAI(
    api_key="<your api key>",
    base_url="https://api.deepseek.com/beta",  # ← Beta 地址
)

# 2. 构造 messages，最后一条是 assistant（前缀）
messages = [
    {"role": "user", "content": "Please write directly in python code to implement quicksort."},
    {"role": "assistant", "content": "```python"}  # ← 这就是前缀
]

# 3. 调用，开启 enable_prefix
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=messages,
    extra_body={"enable_prefix": True},  # ← 开启前缀续写
)

print(response.choices[0].message.content)
```

**逐块解释：**

- **第 1 块（创建客户端）**：注意 `base_url` 末尾是 `/beta`。**为什么？** 因为前缀续写是 Beta 功能，必须走 Beta 通道。如果用普通地址，`enable_prefix` 会被忽略。

- **第 2 块（构造 messages）**：第一条 user 消息是真正的需求"用 Python 实现快排"。第二条 assistant 消息的 content 是 `` ```python ``——这是 Markdown 代码块的开头标记。**为什么这么写？** 因为我们想强制 AI 的回复必须以代码块开头，不能先来一句"好的，这是快排的实现："这种废话。把 `` ```python `` 作为前缀，AI 就只能接着写代码了。

- **第 3 块（调用 API）**：`extra_body={"enable_prefix": True}` 是关键。**为什么用 extra_body？** 因为 `enable_prefix` 是 DeepSeek 自定义的参数，不在 OpenAI 库的标准参数列表里。OpenAI 库的设计是：标准参数直接传，非标准参数塞进 `extra_body`。这样库不会报错，DeepSeek 服务端也能收到。

- **输出结果**：AI 会直接输出快排的 Python 代码（不带前缀 `` ```python ``，因为前缀不算在输出里），最后可能带个 `` ``` `` 闭合代码块。你需要自己把前缀和输出拼起来才是完整内容。

### 4.2 从零跑通指南

#### 第一步：安装依赖

```bash
pip install openai
```

#### 第二步：配置 API Key

参考前面文档，把密钥存到环境变量 `DEEPSEEK_API_KEY`。

#### 第三步：写代码

新建 `prefix_test.py`：

```python
import os
from openai import OpenAI

# 注意：base_url 必须是 beta 地址
client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/beta",
)

# 前缀：强制 AI 以代码块开头
prefix = "```python"
messages = [
    {"role": "user", "content": "用 Python 实现快速排序，直接给代码，不要解释。"},
    {"role": "assistant", "content": prefix},
]

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=messages,
    extra_body={"enable_prefix": True},
    max_tokens=1024,
)

# AI 输出的是前缀之后的内容，需要手动拼上前缀才是完整回复
completion = response.choices[0].message.content
full_reply = prefix + completion

print("=== 完整回复（前缀 + 续写）===")
print(full_reply)
```

#### 第四步：运行

```bash
python prefix_test.py
```

#### 第五步：预期输出

```
=== 完整回复（前缀 + 续写）===
```python
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
```
```

注意：AI 没有说"好的，这是代码"这种开场白，直接就是代码——这就是前缀续写的强制力。

#### 常见错误排查

- **如果 AI 还是自由发挥没接前缀**：检查 `base_url` 是不是 `/beta`，`extra_body` 是不是写对了。
- **如果报错 "extra_body not supported"**：升级 openai 库，`pip install --upgrade openai`。
- **如果输出里没有前缀**：这是正常的！前缀不会出现在 `response.choices[0].message.content` 里，你需要自己拼。

---

## 5. 生产环境视角升华（纯技术向）

### 5.1 适用场景

- **结构化输出强制**：要求 AI 输出必须是 JSON、XML、特定 Markdown 格式时，用前缀锁定开头，避免 AI 加废话开场白（如"好的，这是您要的 JSON："）破坏解析。
- **代码生成**：强制 AI 直接输出代码块，跳过解释性文字，提升下游代码提取的可靠性。
- **模板填充**：业务有固定文案模板（如合同、通知），AI 只负责填充变量部分，前缀锁定模板开头。
- **风格/语气引导**：客服场景强制以"尊敬的客户您好："开头，AI 顺势保持客服语气。
- **多段拼接**：先生成第一段，把第一段作为前缀让 AI 续写第二段，保证连贯性。

### 5.2 避坑指南

- **Beta 稳定性风险**：Beta 通道的功能可能随时调整接口或下线。生产环境强依赖前缀续写时，要做好降级方案（如用 prompt 工程替代：在 user 消息里明确要求"请以 XXX 开头"）。

- **前缀与模型倾向冲突**：如果前缀和模型"想说的"严重冲突（比如前缀是"我同意"，但模型推理认为应该反对），模型可能在续写部分出现逻辑扭曲或自我矛盾。前缀应尽量与任务方向一致。

- **前缀长度限制**：前缀本身也消耗 token，且过长的前缀会压缩续写空间。一般前缀控制在几十 token 以内。

- **流式输出的拼接**：流式模式下，chunk 里只有续写部分，前缀不会出现在任何 chunk 中。前端展示时要手动拼上前缀，否则用户看到的回复会"缺个头"。

- **前缀内容会被计入计费**：前缀作为 messages 的一部分，按 input token 计费。别以为前缀是"免费"的。

- **不要用前缀做安全控制**：想用前缀强制 AI 说特定内容来绕过安全机制，这是不可靠的，模型可能在续写里"找补"回来。安全控制要在 system prompt 和内容审核层做。

### 5.3 最佳实践

**封装一个带前缀的调用函数：**

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/beta",
)

def chat_with_prefix(
    user_msg: str,
    prefix: str = "",
    system_msg: str = None,
    model: str = "deepseek-chat",
    max_tokens: int = 2048,
    temperature: float = 0.7,
    stream: bool = False,
):
    """带前缀续写的对话封装
    
    Args:
        user_msg: 用户问题
        prefix: 强制的回复开头
        system_msg: 可选的系统设定
        ...
    Returns:
        完整回复（前缀 + 续写）
    """
    messages = []
    if system_msg:
        messages.append({"role": "system", "content": system_msg})
    messages.append({"role": "user", "content": user_msg})
    
    use_prefix = bool(prefix)
    if use_prefix:
        messages.append({"role": "assistant", "content": prefix})
    
    kwargs = dict(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=stream,
    )
    if use_prefix:
        kwargs["extra_body"] = {"enable_prefix": True}
    
    if stream:
        # 流式：手动拼前缀 + 累积 chunk
        def gen():
            yield prefix
            resp = client.chat.completions.create(**kwargs)
            for chunk in resp:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        return gen()
    else:
        resp = client.chat.completions.create(**kwargs)
        completion = resp.choices[0].message.content
        return prefix + completion if use_prefix else completion
```

**典型业务用法——强制 JSON 输出：**

```python
# 强制 AI 输出 JSON，避免开场白破坏解析
reply = chat_with_prefix(
    user_msg="分析这段文本的情感，返回 JSON",
    prefix='{"',
    system_msg="你是情感分析助手，只输出 JSON，不要任何解释。",
)
# reply 一定以 {" 开头，可以直接 json.loads
import json
data = json.loads(reply)
```

**降级方案（Beta 不可用时的 fallback）：**

```python
def chat_with_prefix_safe(user_msg, prefix, **kwargs):
    try:
        return chat_with_prefix(user_msg, prefix, **kwargs)
    except Exception as e:
        # Beta 通道异常，降级为普通对话 + prompt 引导
        import logging
        logging.warning(f"Prefix completion failed, fallback: {e}")
        guided_msg = f"{user_msg}\n\n请直接以以下内容开头回复，不要加任何前置说明：\n{prefix}"
        response = client.chat.completions.create(
            model=kwargs.get("model", "deepseek-chat"),
            messages=[{"role": "user", "content": guided_msg}],
            max_tokens=kwargs.get("max_tokens", 2048),
        )
        return response.choices[0].message.content
```
