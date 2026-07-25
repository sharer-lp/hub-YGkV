from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from langchain.tools import tool

@tool
def get_weather(location: str) -> str:
    """Get the weather at a location."""

    if location == "北京":
        return "北京下雪了，明天还是会下雪～"
    if location == "上海":
        return "上海下冰雹了，明天晴天～"
    if location == "武汉":
        return "武汉有雾霾，明天晴天～"

    return f"It's sunny in {location}."


model = ChatOpenAI(
    model="qwen-flash", # 模型的代号
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="sk-9c6195bf91f7435d88ea4b819073c92c"
)
model_with_tools = model.bind_tools([get_weather])

# 人工设定工具
# model_with_tools = model.bind_tools([tool_1], tool_choice="any")  # 所有工具
# model_with_tools = model.bind_tools([tool_1], tool_choice="tool_1") # 强制调用特定的工具

messages = [{"role": "user", "content": "北京和上海、武汉的天气怎么样？ 今天是26年5月1日，总结天气。"}]

# 选择工具，生成params，非stream输出
ai_response = model_with_tools.invoke(messages)
for tool_call in ai_response.tool_calls:
    print(f"Tool: {tool_call['name']}")
    print(f"Args: {tool_call['args']}")
    print("")
messages.append(ai_response)

# 选择工具，生成paramss，tream输出
for chunk in model_with_tools.stream(messages):
    for tool_chunk in chunk.tool_call_chunks:
        if name := tool_chunk.get("name"):
            print(f"Tool: {name}")
        if id_ := tool_chunk.get("id"):
            print(f"ID: {id_}")
        if args := tool_chunk.get("args"):
            print(f"Args: {args}")


# 运行工具
for tool_call in ai_response.tool_calls:
    tool_result = get_weather.invoke(tool_call)
    messages.append(tool_result)

# 汇总得到最终回答
final_response = model_with_tools.invoke(messages)
print(final_response.text)