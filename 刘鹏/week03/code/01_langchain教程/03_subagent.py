from datetime import datetime
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="qwen-flash", # 模型的代号
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="sk-4fedee4ece6541d3b17a7173f0b3c16f"
)

def get_weather_agent(city: str) -> str:
    """Get weather for a given city."""
    query = f"It's always sunny in {city}!"
    result = weather_agent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content

weather_agent = create_agent(
    model=model,
    system_prompt=f"今天{datetime.now()}, 将天气内容总结，并汇总为如下格式：日期、城市、天气。",
)

master_agent = create_agent(
    model=model,
    tools=[get_weather_agent],
    system_prompt="You are a helpful assistant",
)

result = master_agent.invoke(
    {"messages": [{"role": "user", "content": "北京天气怎么样？"}]}
)
print(result["messages"][-1])