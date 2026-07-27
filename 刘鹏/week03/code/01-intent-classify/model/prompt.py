from typing import Union, List

import openai # pip install openai
import pandas as pd
import numpy as np
import jieba
from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer

from config import (
    LLM_OPENAI_API_KEY,
    LLM_OPENAI_SERVER_URL,
    LLM_MODEL_NAME,
    TFIDF_MODEL_PKL_PATH
)

train_data = pd.read_csv('assets/dataset/dataset.csv', sep='\t', header=None)

tfidf, _ = load(TFIDF_MODEL_PKL_PATH)
train_tfidf = tfidf.transform(train_data[0])

client = openai.Client(base_url=LLM_OPENAI_SERVER_URL, api_key=LLM_OPENAI_API_KEY)

# few shot learning
PROMPT_TEMPLATE = '''你是一个意图识别的专家，请结合待选类别和参考例子进行意图分类。
待选类别：{2}

历史参考例子如下：
{1}

待识别的文本为：{0}
只需要输出意图类别（从待选类别中选一个），不要其他输出。'''


def model_for_gpt(request_text: Union[str, List[str]]) -> List[str]:
    classify_result: Union[str, List[str]] = []

    if isinstance(request_text, str):
        tfidf_feat = tfidf.transform([request_text]) # 一个文本
        request_text = [request_text]
    elif isinstance(request_text, list):
        tfidf_feat = tfidf.transform(request_text) # 多个文本
    else:
        raise Exception("格式不支持")

    for query_text, idx in zip(request_text, range(tfidf_feat.shape[0])):
        # 动态提示词
        ids = np.dot(tfidf_feat[idx], train_tfidf.T) # 计算待推理的文本与训练哪些最相似
        top10_index = ids.toarray()[0].argsort()[::-1][:10]

        # 组织为字符串
        # 测试集的每个样本 在训练集中找 最相似的10个作为参考 few shot learning
        # few shot learning 预测的时候，有少量可参考的样本（带标签的）
        dynamic_top10 = ""
        for similar_row in train_data.iloc[top10_index].iterrows():
            dynamic_top10 += similar_row[1][0] + " -> " + similar_row[1][1] + "\n"

        response = client.chat.completions.create(
            # 云端大模型、云端token
            # 本地大模型，本地大模型地址
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "user", "content": PROMPT_TEMPLATE.format(
                    query_text, dynamic_top10, "/".join(list(train_data[1].unique()))
                )},
            ],
            temperature=0,
            max_tokens=64,
        )

        classify_result.append(response.choices[0].message.content)

    return classify_result


# 算法算法项目测试
# 精度、速度、并发、临界情况

# 大模型项目测试
# 精度、速度、并发、临界情况 + 幻觉、提示词攻击/越狱、忠诚度、指令遵循的效果 + 不同大模型的速度

# 服务间调用
# http 居多
# java / c++ 之间 rpc