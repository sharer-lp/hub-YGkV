"""
自我一致性 (Self-Consistency)
==============================
通过对同一个问题多次采样多条推理路径，然后从结果中选择最一致的答案
（多数投票/加权投票），替代单条 CoT 的贪心解码。

步骤：
  1. 用 CoT 提示多次调用 API（可通过 temperature 控制多样性）
  2. 收集每次返回的答案
  3. 用多数投票选出最终答案
"""

from openai import OpenAI

client = OpenAI(
    api_key="sk-e16dfcaa8c7e43908ae4d922f96c4a8f",
    base_url="https://api.deepseek.com")

import re
from collections import Counter


# ---------------------------------------------------------------------------
# 核心：对同一个问题多次采样
# ---------------------------------------------------------------------------
def sample_answers(question: str, system_prompt: str, n: int = 5, temperature: float = 0.7) -> list[str]:
    """对同一个问题采样 n 次，返回 n 个完整输出。"""
    outputs = []
    for i in range(n):
        response = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            temperature=temperature,
            stream=False,
        )
        content = response.choices[0].message.content
        outputs.append(content)
        print(f"  [{i + 1}] {content[:120]}...")
    return outputs


# ---------------------------------------------------------------------------
# 从输出中提取数字答案（简易版）
# ---------------------------------------------------------------------------
def extract_answer(text: str) -> str | None:
    """提取文本中 '答案是 X' 或 'answer is X' 后的数字。"""
    patterns = [
        r"答案是\s*[：:]\s*(\d+)",
        r"答案是\s*(\d+)",
        r"[Aa]nswer\s+is\s+(\d+)",
        r"answer\s*[:：]?\s*(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    # 如果没找到关键词，尝试找最后一行的数字
    lines = text.strip().split("\n")
    for line in reversed(lines):
        nums = re.findall(r"\d+", line)
        if nums:
            return nums[-1]
    return None


# ---------------------------------------------------------------------------
# 示例 1：算术推理 —— 年龄差问题
# ---------------------------------------------------------------------------
def self_consistency_age():
    print("=" * 60)
    print("示例 1：年龄差问题（Self-Consistency）")
    print("=" * 60)

    question = "When I was 6 my sister was half my age. Now I'm 70 how old is my sister?"
    system_prompt = "解决以下问题，逐步推理，最后给出"答案是 X"的格式。"

    outputs = sample_answers(question, system_prompt, n=5, temperature=0.8)

    # 提取答案
    answers = []
    for out in outputs:
        ans = extract_answer(out)
        if ans:
            answers.append(ans)

    # 多数投票
    if answers:
        counter = Counter(answers)
        most_common = counter.most_common()
        final_answer = most_common[0][0]
        print(f"\n所有提取的答案：{answers}")
        print(f"投票结果：{dict(counter)}")
        print(f"最终答案（多数投票）：{final_answer}")
    else:
        print("\n未能从输出中提取数字答案。")
    print()
    return answers


# ---------------------------------------------------------------------------
# 示例 2：数学应用题
# ---------------------------------------------------------------------------
def self_consistency_math():
    print("=" * 60)
    print("示例 2：数学应用题（Self-Consistency）")
    print("=" * 60)

    question = "小明买了一个 35 元的书包和 3 本 12 元的书，付了 100 元，应找回多少钱？"
    system_prompt = "解决以下数学问题，逐步推理，最后给出"答案是 X"的格式。"

    outputs = sample_answers(question, system_prompt, n=5, temperature=0.7)

    answers = []
    for out in outputs:
        ans = extract_answer(out)
        if ans:
            answers.append(ans)

    if answers:
        counter = Counter(answers)
        most_common = counter.most_common()
        final_answer = most_common[0][0]
        print(f"\n所有提取的答案：{answers}")
        print(f"投票结果：{dict(counter)}")
        print(f"最终答案（多数投票）：{final_answer}")
    else:
        print("\n未能从输出中提取数字答案。")
    print()
    return answers


if __name__ == "__main__":
    self_consistency_age()
    self_consistency_math()
