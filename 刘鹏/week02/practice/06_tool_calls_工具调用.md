# DeepSeek API 文档深度讲解：工具调用（Tool Calls / Function Calling）

> 原文出处：https://api-docs.deepseek.com/zh-cn/guides/tool_calls

---

## 1. 一句话概览

这个网页讲的是：**怎么让 AI 不光会"说话"，还会"动手"——你给它一份"工具说明书"，AI 自己判断什么时候该用哪个工具，然后告诉你"帮我调一下查天气的接口"，你照做后把结果喂回给它，它再继续回答。**

打个生活比方：你雇了个聪明的客服，但他没有电脑、查不了库存。于是你给他一张"工具清单"：1号工具查库存、2号工具查物流、3号工具退款。客户来问"我的货到哪了"，客服一看该用2号工具，就递给你一张纸条"请帮我查订单12345的物流"，你查完把结果写在纸条上递回去，客服看了结果再回复客户。AI 工具调用就是这个流程的自动化版本。

---

## 2. 全量专业术语扫盲

### 2.1 工具调用核心概念

- **工具调用（Tool Calls）**：也叫 Function Calling。让 AI 能"调用"外部函数/接口的能力。AI 本身不能联网、不能查数据库、不能算复杂数学，但通过工具调用，它可以"委托"你的程序去干这些事，拿到结果后再继续对话。

- **工具（Tool）**：一个你预先定义好的、AI 可以调用的功能单元。比如"查天气"、"查库存"、"发邮件"。每个工具有名字、说明、参数定义。

- **函数（Function）**：工具的具体实现。在 API 里，工具用函数的形式描述（有函数名、参数、说明）。在你程序里，函数是真正干活的代码。

- **Function Calling（函数调用）**：工具调用的旧称。早期 OpenAI 把这个功能叫 Function Calling，后来改名 Tool Calls，但 API 里很多字段还保留 `function` 字样。

### 2.2 API 参数与流程相关

- **tools 参数**：你传给 API 的"工具说明书"数组。每个元素描述一个工具（函数名、说明、参数）。AI 会读这些说明，判断该不该用、用哪个。

- **tool_calls**：AI 返回的"调用请求"。AI 不直接执行工具（它没这个能力），而是返回一个结构化的调用请求，告诉你"请帮我调用 xxx 函数，参数是 yyy"。你拿到后自己执行。

- **tool_call_id（工具调用 ID）**：每次工具调用的唯一标识。当你把工具执行结果喂回给 AI 时，要用这个 ID 告诉 AI"这是那次调用的结果"，一一对应。

- **tool_choice**：控制 AI 用工具的行为。`auto`（自动决定）、`none`（不用工具）、`required`（必须用工具）、或指定具体工具。

- **strict mode（严格模式）**：DeepSeek 的一个增强功能，开启后 AI 生成的工具调用参数**严格符合**你定义的 schema，不会乱加字段、不会类型错误。对生产环境极有价值。

### 2.3 流程相关

- **多轮工具调用**：一个用户问题可能需要调多个工具，或调一个工具后根据结果再调另一个。整个流程是多轮的：用户提问 → AI 要调工具 → 你执行 → 喂结果 → AI 可能再要调工具 → ... → AI 最终回答。

- **parallel tool calls（并行工具调用）**：AI 一次性请求调用多个工具（比如同时查天气和查日历），你并行执行后一起喂回去。

- **JSON Schema**：一种描述 JSON 结构的标准格式。工具的参数定义用 JSON Schema 写，告诉 AI"参数叫什么、什么类型、必填还是选填"。

### 2.4 编程相关

- **dispatch（分发）**：根据 AI 返回的函数名，找到你程序里对应的函数并执行。通常用一个字典映射 `{"get_weather": get_weather_func}`。

- **serialize/deserialize（序列化/反序列化）**：把函数参数从 JSON 字符串转成 Python 字典，执行后再把结果转回 JSON。

---

## 3. 核心知识与参数全解

### 3.1 工具调用的完整流程

这是理解整个功能的关键，务必记住这个循环：

```
1. 用户提问："北京今天天气怎么样？"
2. 你把问题 + 工具清单一起发给 AI
3. AI 判断：该用 get_weather 工具
   → AI 返回 tool_calls: [{name: "get_weather", arguments: {location: "北京"}}]
   （注意：AI 没有直接回答用户，而是说"我要调工具"）
4. 你的程序：
   a. 解析 tool_calls
   b. 找到 get_weather 函数
   c. 用 {location: "北京"} 调用它
   d. 拿到结果 {temp: 25, weather: "晴"}
5. 你把工具结果作为 role="tool" 的消息，喂回给 AI
   （messages 里现在有：user提问、assistant的tool_calls、tool的结果）
6. AI 看到结果，生成最终回答："北京今天25度，晴天。"
7. 返回给用户
```

**关键认知**：AI 自己**不能**执行任何工具。它只是"说"要调什么，真正执行的是你的代码。AI 是大脑，你的程序是手脚。

### 3.2 所有参数详解

#### 3.2.1 请求参数

| 参数名 | 数据类型 | 默认值 | 作用说明 |
|--------|---------|--------|---------|
| `model` | 字符串 | 无（必填） | 模型名，如 `deepseek-chat`。 |
| `messages` | 数组 | 无（必填） | 对话历史，包括 user、assistant、tool 角色的消息。 |
| `tools` | 数组 | 无 | **工具清单**。每个元素是一个工具定义。不传则 AI 不会调用工具。 |
| `tools[].type` | 字符串 | "function" | 工具类型，目前只有 `"function"`。 |
| `tools[].function` | 字典 | 无 | 函数定义。 |
| `tools[].function.name` | 字符串 | 无 | 函数名，如 `"get_weather"`。AI 用这个名字调用。 |
| `tools[].function.description` | 字符串 | 无 | **函数说明**。**极其重要**——AI 靠这段话判断该不该用这个工具。要写清楚"这个函数干什么、什么时候用"。 |
| `tools[].function.parameters` | 字典 | 无 | 参数的 JSON Schema 定义。描述函数接受什么参数。 |
| `tools[].function.parameters.type` | 字符串 | "object" | 固定为 `"object"`，表示参数是一个对象（键值对）。 |
| `tools[].function.parameters.properties` | 字典 | 无 | 每个参数的定义。key 是参数名，value 是参数的 schema（type、description 等）。 |
| `tools[].function.parameters.required` | 数组 | 无 | 必填参数名列表。 |
| `tool_choice` | 字符串/字典 | "auto" | 控制 AI 用工具的行为。 |
| `tool_choice="auto"` | - | - | AI 自己决定用不用工具（默认）。 |
| `tool_choice="none"` | - | - | 强制不用工具，AI 直接回答。 |
| `tool_choice="required"` | - | - | 强制必须用至少一个工具。 |
| `tool_choice={"type":"function","function":{"name":"xxx"}}` | - | - | 强制用指定的工具。 |
| `strict` | 布尔值 | false | **严格模式**（DeepSeek 增强）。开启后 AI 生成的参数严格符合 schema。 |

#### 3.2.2 返回参数

| 字段名 | 数据类型 | 作用说明 |
|--------|---------|---------|
| `choices[0].message.content` | 字符串/None | 当 AI 决定调工具时，这里通常是 None 或空；当 AI 直接回答时，这里是回答文本。 |
| `choices[0].message.tool_calls` | 数组/None | **AI 的工具调用请求**。每个元素包含 id、函数名、参数。 |
| `choices[0].message.tool_calls[].id` | 字符串 | 这次调用的唯一 ID，喂回结果时要带上。 |
| `choices[0].message.tool_calls[].type` | 字符串 | "function"。 |
| `choices[0].message.tool_calls[].function.name` | 字符串 | 要调用的函数名。 |
| `choices[0].message.tool_calls[].function.arguments` | 字符串 | **参数，是 JSON 字符串**（不是字典！）。需要 `json.loads()` 解析。 |
| `choices[0].finish_reason` | 字符串 | `"tool_calls"` = AI 要调工具（你要执行后继续）；`"stop"` = AI 回答完毕。 |

#### 3.2.3 喂回工具结果时的 messages 结构

执行完工具后，要往 messages 里加两条消息：

1. **assistant 消息**（AI 的 tool_calls 请求）：原样把 AI 返回的 message 加进去，包含 tool_calls 字段。
2. **tool 消息**（工具执行结果）：
   - `role`: `"tool"`
   - `tool_call_id`: 对应的 tool_call.id（**必须一一对应**）
   - `content`: 工具执行结果的字符串（通常是 JSON 字符串）

### 3.3 strict 严格模式详解

普通模式下，AI 生成的参数可能：
- 多出你没定义的字段
- 字段类型不对（要 string 给了 number）
- 必填字段缺失

strict 模式开启后，DeepSeek 在生成时强制约束参数严格符合 schema，杜绝上述问题。生产环境强烈建议开启。

**开启方式**（以 DeepSeek 实际支持为准，参考官网）：
- 在 function 定义里加 `"strict": true`
- 或在请求顶层传 `strict: true`

---

## 4. 代码示例剖析与环境配置指南

### 4.1 代码剖析

网页给出的工具调用示例（整理为完整可运行版）：

```python
import json
from openai import OpenAI

client = OpenAI(
    api_key="<your api key>",
    base_url="https://api.deepseek.com",
)

# 1. 定义工具清单
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "城市名，如'北京'、'上海'"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

# 2. 真正干活的函数
def get_weather(location: str) -> str:
    # 实际项目里这里调真实天气 API
    weather_data = {"北京": "晴 25度", "上海": "多云 28度"}
    return json.dumps({"location": location, "weather": weather_data.get(location, "未知")})

# 3. 第一轮：用户提问
messages = [
    {"role": "user", "content": "北京今天天气怎么样？"}
]

# 4. 第一次调 API，AI 决定调工具
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=messages,
    tools=tools,
)

# 5. 检查 AI 是否要调工具
assistant_message = response.choices[0].message
if assistant_message.tool_calls:
    # 把 AI 的 tool_calls 请求加入历史
    messages.append(assistant_message)
    
    # 6. 执行每个工具调用
    for tool_call in assistant_message.tool_calls:
        func_name = tool_call.function.name
        func_args = json.loads(tool_call.function.arguments)  # 解析参数
        
        print(f"AI 要调用: {func_name}({func_args})")
        
        # 分发到对应函数
        if func_name == "get_weather":
            result = get_weather(**func_args)
        else:
            result = json.dumps({"error": f"未知函数: {func_name}"})
        
        print(f"工具结果: {result}")
        
        # 7. 把工具结果作为 tool 消息加入历史
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result,
        })
    
    # 8. 第二次调 API，AI 根据工具结果生成最终回答
    final_response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        tools=tools,
    )
    print("最终回答:", final_response.choices[0].message.content)
else:
    # AI 没要调工具，直接回答了
    print("直接回答:", assistant_message.content)
```

**逐块解释：**

- **第 1 块（定义工具清单）**：`tools` 数组告诉 AI"你手上有这些工具可用"。`description` 极其重要——AI 靠它判断什么时候用这个工具。要写得具体、明确。

- **第 2 块（定义真实函数）**：这是你程序里真正干活的代码。AI 不知道也不关心它的实现，只关心"调用它能拿到什么结果"。

- **第 3~4 块（第一次调 API）**：把用户问题 + 工具清单一起发给 AI。AI 会判断：该不该用工具？用哪个？参数是什么？

- **第 5 块（检查 tool_calls）**：**关键判断**。如果 `assistant_message.tool_calls` 非空，说明 AI 要调工具；否则 AI 直接回答了。这两种情况要分开处理。

- **第 6 块（执行工具）**：
  - `json.loads(tool_call.function.arguments)` 把参数从字符串转成字典。**注意 arguments 是字符串不是字典**，新手必踩坑。
  - 用 if-else 或字典映射把函数名分发到对应函数。
  - `get_weather(**func_args)` 用 `**` 把字典解包成关键字参数。

- **第 7 块（喂回结果）**：把工具结果作为 `role="tool"` 的消息加入 messages。**`tool_call_id` 必须对应**，否则 AI 不知道这个结果是哪次调用的。

- **第 8 块（第二次调 API）**：带着完整历史（用户问题 + AI 的工具调用请求 + 工具结果）再调一次 API。AI 看到结果后生成最终自然语言回答。

### 4.2 从零跑通指南

#### 第一步：安装依赖

```bash
pip install openai
```

#### 第二步：配置 API Key

参考前面文档，设置 `DEEPSEEK_API_KEY` 环境变量。

#### 第三步：写代码

新建 `tool_calls_test.py`，把上面的完整代码贴进去，把 `<your api key>` 改成 `os.environ.get("DEEPSEEK_API_KEY")`。

#### 第四步：运行

```bash
python tool_calls_test.py
```

#### 第五步：预期输出

```
AI 要调用: get_weather({'location': '北京'})
工具结果: {"location": "北京", "weather": "晴 25度"}
最终回答: 北京今天天气晴朗，气温25度。
```

#### 常见错误排查

- **`arguments` 是字符串不是字典**：直接 `tool_call.function.arguments["location"]` 会报错。必须先 `json.loads()`。
- **`tool_call_id` 不匹配**：喂回结果时 id 必须严格对应，否则 API 报错。
- **AI 反复调同一个工具不收敛**：可能是工具 description 写得不清，或工具结果格式 AI 看不懂。加 max_iterations 限制防止死循环。
- **AI 不调工具直接瞎编**：description 写得太模糊，AI 不知道该用。或 tool_choice 设成了 none。
- **函数名拼错**：tools 里定义的 name 必须和分发逻辑里的判断完全一致。

---

## 5. 生产环境视角升华（纯技术向）

### 5.1 适用场景

- **信息检索增强（RAG）**：AI 调用搜索工具、数据库查询工具，获取实时或私有知识。
- **外部 API 集成**：AI 调用发邮件、发短信、创建日历事件、提交订单等业务 API。
- **代码执行**：AI 调用代码解释器工具，执行计算、数据分析、生成图表。
- **多步推理 Agent**：复杂任务拆解成多步，每步调不同工具，形成 Agent 工作流。
- **数据库操作**：Text-to-SQL，AI 生成 SQL 并调用查询工具，返回结果。
- **自动化运维**：AI 调用重启服务、查看日志、扩容等运维工具。

### 5.2 避坑指南

- **死循环风险**：AI 可能反复调工具不收敛（比如工具一直返回错误，AI 一直重试）。**必须设 max_iterations 上限**（如 5 次），超过就强制返回"无法完成"。

- **工具执行超时**：外部 API 可能慢或挂掉。每个工具调用要设超时，超时后返回结构化错误给 AI，让它决定怎么办。

- **参数注入攻击**：AI 生成的参数可能包含恶意内容（用户诱导 AI 调危险工具）。**所有工具参数都要做校验和清洗**，特别是涉及 SQL、shell、文件路径的。

- **权限控制**：不同用户能用的工具不同。工具执行前要校验当前用户权限，别让普通用户调到管理员工具。

- **工具描述质量**：description 写不好，AI 就用不对。要写清楚：干什么、什么时候用、什么时候不用、返回什么。可以给 few-shot 示例。

- **并行调用的结果对应**：AI 一次可能调多个工具，结果要按 tool_call_id 严格对应，别搞混。

- **成本失控**：复杂任务可能调几十次 API。要监控单次任务的 API 调用次数和 token 消耗，设上限。

- **错误处理**：工具执行失败时，别直接抛异常给用户。返回结构化错误给 AI，让它优雅降级（"抱歉，查询失败，建议您..."）。

- **幂等性**：工具可能被重复调用（AI 重试）。涉及写操作的工具（下单、发邮件）要保证幂等，防止重复执行。

- **审计日志**：生产环境必须记录每次工具调用的：谁、什么时候、调了什么、参数、结果。出问题能追溯。

### 5.3 最佳实践

**生产级工具调用框架：**

```python
import os
import json
import logging
from openai import OpenAI
from typing import Callable, Any

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

logger = logging.getLogger(__name__)

class ToolRegistry:
    """工具注册中心"""
    def __init__(self):
        self._tools = {}  # name -> (schema, func)
    
    def register(self, name: str, description: str, parameters: dict, func: Callable):
        self._tools[name] = ({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            }
        }, func)
    
    def get_schemas(self) -> list:
        return [schema for schema, _ in self._tools.values()]
    
    def execute(self, name: str, arguments: dict) -> str:
        if name not in self._tools:
            return json.dumps({"error": f"Unknown tool: {name}"})
        _, func = self._tools[name]
        try:
            result = func(**arguments)
            return json.dumps(result, ensure_ascii=False, default=str)
        except Exception as e:
            logger.exception(f"Tool {name} execution failed")
            return json.dumps({"error": str(e)})

def run_agent(
    user_message: str,
    registry: ToolRegistry,
    system_prompt: str = None,
    max_iterations: int = 5,
) -> str:
    """生产级 Agent 循环，带迭代上限和日志"""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})
    
    tools = registry.get_schemas()
    
    for i in range(max_iterations):
        logger.info(f"Agent iteration {i+1}")
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        
        msg = response.choices[0].message
        finish_reason = response.choices[0].finish_reason
        
        if not msg.tool_calls:
            # AI 给出最终回答
            return msg.content or ""
        
        # AI 要调工具
        messages.append(msg)
        
        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                args = {}
            
            logger.info(f"Tool call: {name}({args})")
            result = registry.execute(name, args)
            logger.info(f"Tool result: {result[:200]}")
            
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })
    
    # 超过最大迭代
    logger.warning(f"Agent exceeded max_iterations={max_iterations}")
    return "抱歉，处理这个请求需要的步骤太多，暂时无法完成。请简化您的问题。"

# 注册工具
registry = ToolRegistry()
registry.register(
    name="get_weather",
    description="获取指定城市的实时天气。当用户询问天气、是否需要带伞、穿什么衣服时使用。",
    parameters={
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "城市名"}
        },
        "required": ["location"]
    },
    func=lambda location: {"location": location, "temp": 25, "weather": "晴"},
)

# 使用
answer = run_agent(
    "北京今天天气怎么样？适合出门吗？",
    registry,
    system_prompt="你是天气助手，可以查询天气并给出建议。",
)
print(answer)
```

**监控指标：**

- 单次任务平均工具调用次数、平均 API 调用次数
- 工具调用成功率、各工具的失败率
- 工具执行延迟 P50/P95/P99
- 达到 max_iterations 上限的任务比例
- 各工具被调用的频率分布（识别高频工具，考虑缓存或优化）
- 单次任务总 token 消耗和成本
