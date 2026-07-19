"""
生成知识提示 (Generate Knowledge Prompting)
==========================================
先让模型生成与问题相关的背景知识/事实，再基于这些知识回答问题。
两步过程：
  1. Knowledge Generation：让模型生成关于主题的知识陈述
  2. Knowledge Integration：将生成的知识作为上下文来回答问题

这种方法特别适合需要常识推理、专业知识或最新信息的问答任务。
"""

from openai import OpenAI

client = OpenAI(
    api_key="sk-e16dfcaa8c7e43908ae4d922f96c4a8f",
    base_url="https://api.deepseek.com")


# ---------------------------------------------------------------------------
# 两步式工具函数：生成知识 → 回答问题
# ---------------------------------------------------------------------------
def generate_knowledge(topic: str) -> str:
    """第一步：生成与问题相关的背景知识。"""
    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {"role": "system", "content": "你是一个知识库。生成与用户问题相关的背景知识，列出关键事实。只输出知识内容，不要回答问题。"},
            {"role": "user", "content": topic},
        ],
        stream=False,
    )
    knowledge = response.choices[0].message.content
    print(f"[生成的知识]\n{knowledge[:200]}...\n")
    return knowledge


def answer_with_knowledge(question: str, knowledge: str) -> str:
    """第二步：基于生成的知识来回答问题。"""
    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {"role": "system", "content": "使用下面提供的背景知识来回答用户问题。如果知识不足以回答，请说明。"},
            {"role": "assistant", "content": f"背景知识：\n{knowledge}"},
            {"role": "user", "content": question},
        ],
        stream=False,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# 示例 1：常识推理 —— 企鹅为什么不会飞？
# ---------------------------------------------------------------------------
def knowledge_penguin():
    print("=" * 60)
    print("示例 1：常识推理 —— 企鹅为什么不会飞？")
    print("=" * 60)

    question = "企鹅为什么不会飞？"
    knowledge = generate_knowledge(f"关于「{question}」的相关生物学知识。列出企鹅的演化历史、身体结构、生存环境等关键事实。")
    answer = answer_with_knowledge(question, knowledge)
    print(f"[最终回答]\n{answer}\n")
    return answer


# ---------------------------------------------------------------------------
# 示例 2：科学推理 —— 为什么海水是咸的？
# ---------------------------------------------------------------------------
def knowledge_seawater():
    print("=" * 60)
    print("示例 2：科学推理 —— 为什么海水是咸的？")
    print("=" * 60)

    question = "为什么海水是咸的？"
    knowledge = generate_knowledge(f"关于「{question}」的地质学和化学知识。包括岩石风化、矿物质溶解、水循环等。")
    answer = answer_with_knowledge(question, knowledge)
    print(f"[最终回答]\n{answer}\n")
    return answer


# ---------------------------------------------------------------------------
# 示例 3：专业领域 —— 推荐系统冷启动问题
# ---------------------------------------------------------------------------
def knowledge_cold_start():
    print("=" * 60)
    print("示例 3：专业领域 —— 推荐系统冷启动问题")
    print("=" * 60)

    question = "推荐系统中的冷启动问题有哪些常见的解决方案？"
    knowledge = generate_knowledge(
        f"关于推荐系统冷启动问题的知识。"
        f"包括用户冷启动、物品冷启动、系统冷启动三种场景的定义和特点。"
    )
    answer = answer_with_knowledge(question, knowledge)
    print(f"[最终回答]\n{answer}\n")
    return answer


# ---------------------------------------------------------------------------
# 对比：直接回答（无知识生成）vs 有知识生成
# ---------------------------------------------------------------------------
def compare_direct_vs_knowledge():
    print("=" * 60)
    print("对比实验：直接回答 vs 生成知识后回答")
    print("=" * 60)

    question = "袋鼠的尾巴有什么作用？"

    # 直接回答
    print("--- 直接回答 ---")
    direct = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {"role": "system", "content": "回答用户问题。"},
            {"role": "user", "content": question},
        ],
        stream=False,
    )
    print(f"[直接回答]\n{direct.choices[0].message.content}\n")

    # 先生成知识
    print("--- 生成知识后回答 ---")
    knowledge = generate_knowledge(f"关于「{question}」的动物学知识。")
    with_knowledge = answer_with_knowledge(question, knowledge)
    print(f"[有知识回答]\n{with_knowledge}\n")


if __name__ == "__main__":
    knowledge_penguin()
    knowledge_seawater()
    knowledge_cold_start()
    compare_direct_vs_knowledge()
