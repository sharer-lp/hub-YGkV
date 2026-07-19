"""
流式输出 — 逐 chunk 展示模型回复（含 thinking 过程）
"""
"""
目录：
1. 核心概念：什么是 delta？(基础知识与扩展)
2. 流式输出到底是怎么结束的？(eos, max_token, stop 详解)
3. 控制台打字机效果的底层原理（澄清你的误解）
4. 补充知识：SSE 协议与流式异常处理
5. 综合实战：带有结束状态判断和异常处理的高级流式代码
"""

# ==============================================================================
# 1. 核心概念：什么是 delta？(基础知识与扩展)
# ==============================================================================
"""
【基础知识】
在 OpenAI 兼容的 API 中，非流式和流式返回的数据结构是不同的：
- 非流式: response.choices[0].message (包含完整的角色和完整的内容)
- 流式  : chunk.choices[0].delta   (只包含"增量"内容)

delta 直译为"增量"或"差值"。大模型在流式模式下，并不是等全想好了再发给你，
而是每生成几个字（一个 Token），就把这几个字打包成一个 chunk 发给你。
这个 chunk 里的 delta 字段，装的仅仅是"刚刚新生成的这几个字"。

【扩展信息：delta 的生命周期】
在一个完整的流式对话中，delta 对象会经历以下阶段：

第 1 个 chunk: delta 通常包含 role，content 为空。
    例如: {"role": "assistant", "content": ""}  (告诉你接下来的话是 AI 说的)

第 2 到 N 个 chunk: delta 只包含 content，role 消失。
    例如: {"content": "你好"}
    例如: {"content": "，世界"}

最后一个 chunk: delta 通常是空的 (content 为 None)，但此时 choice 中会出现
              finish_reason 字段，告诉你对话为什么结束了。
    例如: {"content": null}, finish_reason="stop"

【为什么用 delta？】
节省带宽和降低首字延迟。如果每次都发完整的 message，随着内容越来越长，
每个 chunk 的体积会越来越大。用 delta 只传"变化的部分"，网络传输效率最高。
"""

# ==============================================================================
# 2. 流式输出到底是怎么结束的？(eos, max_token, stop 详解)
# ==============================================================================
"""
流式循环 (for chunk in stream) 总有结束的时候，结束的原因记录在
choice.finish_reason 字段中。常见的结束方式有三种（对应你问的三个概念）：

1. 正常结束：finish_reason = "stop"
   - 对应你说的 eos (End of Sequence，序列结束符)。
   - 这是模型自己判断"话说完了"，它在生成时输出了一个特殊的内部结束符 <eos>。
   - 这是最理想的结束状态。

2. 达到最大长度限制：finish_reason = "length"
   - 对应你说的 max_token。
   - 因为你设置了 max_tokens=200，当模型生成到 200 个 token 时，哪怕它话没说完，
     API 也会强行掐断流，此时 finish_reason 就是 "length"。
   - 遇到这种情况，你需要知道回答是不完整的。

3. 遇到停止词：finish_reason = "stop" (但触发条件不同)
   - 对应你说的 stop。
   - 你可以在 API 中传入一个 stop=["\n\n"] 数组。
   - 如果模型生成的内容中命中了你设定的 stop 词，API 会立即停止生成。
   - 虽然原因也是 "stop"，但在 response 的结构中会有细微差别（或者你可以通过
     内容是否包含 stop 词来推断）。

4. 补充：触发安全审查：finish_reason = "content_filter"
   - 模型的回答触发了平台的安全合规策略，被中途拦截。

【如何捕获？】
在循环中，你需要检查 chunk.choices[0].finish_reason 是否不为 None。
"""

# ==============================================================================
# 3. 控制台打字机效果的底层原理（澄清你的误解）
# ==============================================================================
"""
【你的误解】
你提到："直接打印 print(full_content)，每次输出内容都会包含之前的内容，
所以输出的信息都是存在重复的内容。"
你说的完全正确！如果每次循环里都 print(full_content)，控制台确实会打印出
越来越长、包含大量重复的整段文本。

【真相揭晓：打字机效果靠的是 delta，不是 full_content】
在之前给的流式代码中，打字机效果是由这行代码实现的：
    print(delta.content, end="", flush=True)

拆解这行代码：
1. delta.content: 这里打印的是"当前新增的碎片"，比如 "你"，下一次是 "好"。
   它绝不会包含之前的内容。

2. end="": 这是 Python print 函数的参数。默认 print 会在末尾加一个换行符(\n)。
   设置 end="" 表示不换行。这样光标就会停在 "你" 的后面，等待下一次打印。

3. flush=True: 默认情况下，print 会把内容放到缓冲区，等缓冲区满了才显示在屏幕上。
   flush=True 强制立即把内容刷新到屏幕上。

【工作流程演示】
假设模型生成 "你好"：
- 循环第1次: delta.content="你"
  print("你", end="", flush=True) -> 屏幕显示: 你 (光标紧跟其后)
  后台执行: full_content += "你" (此时 full_content = "你")

- 循环第2次: delta.content="好"
  print("好", end="", flush=True) -> 屏幕显示: 你好 (光标紧跟其后)
  后台执行: full_content += "好" (此时 full_content = "你好")

- 循环结束后:
  print(full_content) -> 这是在最后为了确认完整内容才打印的，只打印一次。
"""

# ==============================================================================
# 4. 补充知识：SSE 协议与流式异常处理
# ==============================================================================
"""
【SSE (Server-Sent Events) 协议】
流式输出的底层依赖 HTTP 的 SSE 协议。简单来说，服务器不是一次性返回 JSON，
而是保持连接不断开，持续向客户端推送 "data: {...}\n\n" 格式的文本块。
OpenAI SDK 帮你把这些复杂的网络解析封装成了 Python 的迭代器 (for chunk in stream)。

【流式异常处理】
流式请求容易遇到网络波动（比如生成到一半断网了）。此时会抛出 APIConnectionError。
因为你已经收到了一部分内容，所以不能简单地直接崩溃。实际开发中需要 try...except，
甚至结合断点续传（高级功能），但最基础的是要捕获异常，避免程序崩溃。
"""

from openai import OpenAI

client = OpenAI(
    api_key="sk-879e6628fec6417cbfd6b69c3e4d6ac0",
    base_url="https://api.deepseek.com",
)

stream = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "你好，帮我介绍机器学习"},
    ],
    stream=True,
    reasoning_effort="high",
    extra_body={"thinking": {"type": "enabled"}},
)

full_content = ""
reasoning_content = ""

for chunk in stream:
    choice = chunk.choices[0] if chunk.choices else None
    if choice is None:
        # 最后一个 chunk 可能只包含 usage 信息
        if hasattr(chunk, "usage") and chunk.usage:
            print(f"\n--- Token 用量 ---")
            print(f"  prompt_tokens:     {chunk.usage.prompt_tokens}")
            print(f"  completion_tokens: {chunk.usage.completion_tokens}")
            print(f"  total_tokens:      {chunk.usage.total_tokens}")
        continue

    delta = choice.delta

    # reasoning 内容（若模型返回）
    if hasattr(delta, "reasoning_content") and delta.reasoning_content:
        reasoning_content += delta.reasoning_content
        print(delta.reasoning_content, end="", flush=True)

    # 常规内容
    if delta.content:
        full_content += delta.content
        print(delta.content, end="", flush=True)

print("\n" + "=" * 50)
print("完整回复：")
print(full_content)
