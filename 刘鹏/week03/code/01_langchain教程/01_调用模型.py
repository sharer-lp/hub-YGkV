from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage, HumanMessage
# AIMessage 大模型对应的对话内容
# HumanMessage 用户的对话内容

# 连接openai 类型大模型的客户端
llm = ChatOpenAI(
    model="qwen-flash", # 模型的代号
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="sk-4fedee4ece6541d3b17a7173f0b3c16f"
)

response = llm.invoke("你好") # 默认输入user内容
print(response.content)

msg = HumanMessage(content="你好", name="Lance")
messages = [msg, msg]
response = llm.invoke(messages)
print(response.content)



messages = [AIMessage(content=f"So you said you were researching ocean mammals?", name="Model")]
messages.append(HumanMessage(content=f"Yes, that's right.",name="Lance"))
messages.append(AIMessage(content=f"Great, what would you like to learn about.", name="Model"))
messages.append(HumanMessage(content=f"I want to learn about the best place to see Orcas in the US.", name="Lance"))

for m in messages:
    m.pretty_print()