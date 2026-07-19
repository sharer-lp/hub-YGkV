"""
少样本提示 (Few-shot Prompting)
===============================
在提示中给模型提供几个输入-输出示例（demonstrations），
让模型从示例中学习任务格式和模式，再完成新的输入。

关键点：
  - 示例数量通常 2-5 个效果较好
  - 示例要覆盖任务的变化模式
  - 示例格式要和期望输出一致
"""

from openai import OpenAI

client = OpenAI(
    api_key="sk-e16dfcaa8c7e43908ae4d922f96c4a8f",
    base_url="https://api.deepseek.com")


# ---------------------------------------------------------------------------
# 示例 1：少样本情感分类
# ---------------------------------------------------------------------------
def few_shot_sentiment():
    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {"role": "system", "content": "根据示例格式，判断句子情感为「正面」或「负面」。"},
            {"role": "user", "content": """商品：这款耳机音质太好听了，低音效果很棒！
情感：正面

商品：快递等了10天都没到，太失望了。
情感：负面

商品：屏幕清晰度一般，但价格还算合理。
情感：正面

商品：刚用三天就坏了，质量太差了。
情感："""},
        ],
        stream=False,
    )
    print("=== 少样本：情感分类 ===")
    print(f"输入：刚用三天就坏了，质量太差了。")
    print(f"输出：{response.choices[0].message.content}\n")
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# 示例 2：少样本 —— 生成反义词
# ---------------------------------------------------------------------------
def few_shot_antonym():
    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {"role": "user", "content": """好的 -> 坏
大 -> 小
快 -> 慢
热 ->
"""},
        ],
        stream=False,
    )
    print("=== 少样本：反义词生成 ===")
    print(f"输入：热")
    print(f"输出：{response.choices[0].message.content}\n")
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# 示例 3：少样本 —— 数学应用题（与 Self-Consistency 配合的基础）
# ---------------------------------------------------------------------------
def few_shot_math():
    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {"role": "system", "content": "解决数学问题，给出推理过程和答案。"},
            {"role": "user", "content": """Q: 小明有 5 个苹果，小红给了小明 3 个苹果，现在小明有多少个苹果？
A: 小明原来有 5 个苹果，小红给了他 3 个，所以现在有 5 + 3 = 8 个苹果。答案是 8。

Q: 教室里有 12 个学生，走了 4 个，又来了 6 个，现在教室有多少个学生？
A: 教室里原来有 12 个学生，走了 4 个剩下 12 - 4 = 8 个，又来了 6 个，现在有 8 + 6 = 14 个学生。答案是 14。

Q: 一本书 50 元，一个笔记本比书便宜 35 元，买一本书和一个笔记本一共多少钱？
A:"""},
        ],
        stream=False,
    )
    print("=== 少样本：数学应用题 ===")
    print("输入：一本书 50 元，一个笔记本比书便宜 35 元，买一本书和一个笔记本一共多少钱？")
    print(f"输出：{response.choices[0].message.content}\n")
    return response.choices[0].message.content


if __name__ == "__main__":
    few_shot_sentiment()
    few_shot_antonym()
    few_shot_math()
