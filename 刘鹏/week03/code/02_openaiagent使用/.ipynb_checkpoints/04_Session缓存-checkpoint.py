import time
import os
os.environ["OPENAI_API_KEY"] = "sk-78cc4e9ac8f44efdb207b7232e1ae6d8"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

import asyncio
from agents import Agent, Runner, SQLiteSession
from agents import set_default_openai_api, set_tracing_disabled
set_default_openai_api("chat_completions")
set_tracing_disabled(True)

session = SQLiteSession("conversation_123")

async def main():
    agent = Agent(
        name="Assistant",
        model="qwen-max",
        instructions="Reply very concisely.",
    )
    
    result = await Runner.run(
        agent,
        "我叫王五，请给我讲笑话",
        session=session
    )
    print(result.final_output)

    result = await Runner.run(
        agent,
        "帮我计算100 * 1000",
        session=session
    )
    print(result.final_output)

    result = await Runner.run(
        agent,
        "我是谁？",
        session=session
    )
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())