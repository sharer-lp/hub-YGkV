"""
思维链提示 (Chain-of-Thought Prompting / CoT)
=============================================
在提示中引导模型逐步推理（中间步骤），而不是直接跳到答案。
显著提升需要多步推理的任务（数学、逻辑、常识推理）的准确性。

两种方式：
  1. Few-shot CoT：给出带推理过程的示例
  2. Zero-shot CoT：在提示末尾加"Let's think step by step"（逐步思考）
"""

from openai import OpenAI

client = OpenAI(
    api_key="sk-e16dfcaa8c7e43908ae4d922f96c4a8f",
    base_url="https://api.deepseek.com")


# ---------------------------------------------------------------------------
# 示例 1：Zero-shot CoT —— 直接让模型逐步思考
# ---------------------------------------------------------------------------
def zero_shot_cot():
    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {"role": "system", "content": "请逐步推理，最后给出答案。"},
            {"role": "user", "content": "一个池塘里有一片荷叶，每天荷叶面积翻倍。第30天荷叶覆盖了整个池塘。请问第几天荷叶覆盖了池塘的一半？请逐步推理。"},
        ],
        stream=False,
    )
    print("=== Zero-shot CoT：池塘荷叶问题 ===")
    print("问题：一个池塘里有一片荷叶，每天荷叶面积翻倍。第30天覆盖整个池塘，第几天覆盖一半？")
    print(f"回答：{response.choices[0].message.content}\n")
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# 示例 2：Few-shot CoT —— 带推理过程的示例
# ---------------------------------------------------------------------------
def few_shot_cot():
    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {"role": "system", "content": "解决数学问题，逐步推理后给出答案。"},
            {"role": "user", "content": """Q: 商店有 25 个篮球，卖出了 13 个，又进货了 20 个。现在商店有多少个篮球？
A: 商店原来有 25 个篮球，卖出 13 个后剩下 25 - 13 = 12 个。又进货 20 个，现在有 12 + 20 = 32 个篮球。答案是 32。

Q: 小明有 45 元钱，买了 3 支笔每支 8 元，还剩多少钱？
A: 小明有 45 元。3 支笔每支 8 元，总共花了 3 × 8 = 24 元。剩余 45 - 24 = 21 元。答案是 21。

Q: 一辆火车从 A 站出发，先载了 40 人，到 B 站下了 12 人上了 18 人，到 C 站下了 25 人上了 8 人。火车到 D 站时车上有多少人？
A:"""},
        ],
        stream=False,
    )
    print("=== Few-shot CoT：火车乘客问题 ===")
    print("问题：一辆火车从 A 站出发，先载了 40 人，到 B 站下了 12 人上了 18 人，到 C 站下了 25 人上了 8 人。到 D 站时车上有多少人？")
    print(f"回答：{response.choices[0].message.content}\n")
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# 示例 3：常识推理 —— CoT 解决逻辑陷阱题
# ---------------------------------------------------------------------------
def commonsense_cot():
    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {"role": "system", "content": "请逐步推理，最后给出答案。"},
            {"role": "user", "content": "我 6 岁的时候妹妹是我年龄的一半。现在 70 岁了，妹妹多少岁？请在推理中注意年龄差是不变的，最后给出答案。"},
        ],
        stream=False,
    )
    print("=== Zero-shot CoT：年龄陷阱题 ===")
    print("问题：我 6 岁时妹妹是我年龄的一半。现在 70 岁了，妹妹多少岁？")
    print(f"回答：{response.choices[0].message.content}\n")
    return response.choices[0].message.content


if __name__ == "__main__":
    zero_shot_cot()
    few_shot_cot()
    commonsense_cot()
