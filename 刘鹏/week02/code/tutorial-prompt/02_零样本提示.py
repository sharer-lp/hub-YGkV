"""
零样本提示 (Zero-shot Prompting)
===============================
零样本提示是最基础的提示方式：不给任何示例，直接让模型完成任务。
关键在于提供清晰、明确的指令。

适用场景：模型已有相关知识储备的通用任务。
"""

from openai import OpenAI

client = OpenAI(
    api_key="sk-e16dfcaa8c7e43908ae4d922f96c4a8f",
    base_url="https://api.deepseek.com")


# ---------------------------------------------------------------------------
# 示例 1：文本分类 —— 零样本情感分析
# ---------------------------------------------------------------------------
def zero_shot_classification():
    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {"role": "system", "content": "你是一个文本分类助手。请将用户输入的句子分类为「正面」「负面」或「中性」，只输出分类结果。"},
            {"role": "user", "content": "这家酒店的早餐非常丰盛，服务也很周到！"},
        ],
        stream=False,
    )
    print("=== 零样本：文本分类 ===")
    print(f"输入：这家酒店的早餐非常丰盛，服务也很周到！")
    print(f"输出：{response.choices[0].message.content}\n")
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# 示例 2：零样本翻译
# ---------------------------------------------------------------------------
def zero_shot_translation():
    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {"role": "system", "content": "你是一个翻译助手。将用户输入的中文翻译成英文，只输出翻译结果。"},
            {"role": "user", "content": "人工智能正在改变世界的每个角落。"},
        ],
        stream=False,
    )
    print("=== 零样本：翻译 ===")
    print(f"输入：人工智能正在改变世界的每个角落。")
    print(f"输出：{response.choices[0].message.content}\n")
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# 示例 3：零样本摘要
# ---------------------------------------------------------------------------
def zero_shot_summarization():
    text = (
        "机器学习是人工智能的一个分支领域，它使计算机能够从数据中学习模式而无需显式编程。"
        "监督学习使用标注数据训练模型，无监督学习则发现未标注数据中的隐藏结构。"
        "深度学习作为机器学习的子集，使用多层神经网络处理复杂任务如图像识别和自然语言处理。"
        "近年来，大语言模型（LLM）在文本生成、问答和代码编写等方面展现出惊人能力。"
    )
    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {"role": "system", "content": "用一句话概括用户提供的文本。"},
            {"role": "user", "content": text},
        ],
        stream=False,
    )
    print("=== 零样本：文本摘要 ===")
    print(f"输入：{text[:40]}...")
    print(f"输出：{response.choices[0].message.content}\n")
    return response.choices[0].message.content


if __name__ == "__main__":
    zero_shot_classification()
    zero_shot_translation()
    zero_shot_summarization()
