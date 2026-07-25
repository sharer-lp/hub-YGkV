import pandas as pd
import jieba
from gensim.models import Word2Vec # 训练、加载 词向量

# 加载数据集
dataset = pd.read_csv("../Week01/dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()

# 1. 使用jieba对文本进行分词
# 这一步将每句话分解为一个词语列表。
# 这种列表的列表格式正是Word2Vec所期望的输入格式。
tokenized_sentences = [list(jieba.cut(text)) for text in texts]

# 2. 训练Word2Vec模型
# 我们将使用Skip-gram模型（sg=1）来从目标词预测上下文词。
# 向量维度（vector_size）和窗口大小（window）是重要的超参数。
# min_count=1 表示我们不忽略任何词，即使它只出现一次。
model = Word2Vec( # cpu 训练
    sentences=tokenized_sentences,
    vector_size=100,      # 词向量的维度
    window=5,             # 当前词与预测词之间的最大距离
    min_count=1,          # 忽略总频率低于此值的词

    sg=1                  # 1表示Skip-gram，0表示CBOW
)

# 你可以保存训练好的模型，以便将来直接使用，无需重新训练。
model.save("word2vec_model.bin") # 使用的时候， 单词编码结果 不变
# 要加载已保存的模型：
# model = Word2Vec.load("word2vec_model.bin")

# 3. 使用已训练好的模型
print("--- 使用 Word2Vec 模型 ---")

# a) 查找与给定词最相似的词
# 这会返回一个 (词语, 相似度得分) 的元组列表。
similar_words = model.wv.most_similar("北京", topn=5)
print("与 '北京' 最相似的词：")
for word, score in similar_words:
    print(f"  {word}: {score:.4f}")

print("\n--------------------------")

# b) 计算两个词之间的相似度
similarity_score = model.wv.similarity("天气", "下雨")
print(f"'天气' 和 '下雨' 之间的相似度: {similarity_score:.4f}")

similarity_score_2 = model.wv.similarity("导航", "北京")
print(f"'导航' 和 '北京' 之间的相似度: {similarity_score_2:.4f}")

# 你也可以获取特定词的向量。
vector = model.wv['导航']
print("\n'导航' 的向量：")
print(vector)
print(f"向量大小: {vector.shape}")