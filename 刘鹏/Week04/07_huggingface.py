import torch

# 模型：分词器， 模型结构（权重）
# Auto 自动推理 模型权重和配置，加载对应的模块
# Tokenizer 分词器： 类似jieba 每个bert / qwen /ds 都有自己的 分词器
# ModelForMaskedLM 加载一个模型，修改输出结构用于MLM
from transformers import AutoTokenizer, AutoModelForMaskedLM
# transformers 语言建模的库，模型结构

# 所有bert 、gpt （qwen、ds、chatgpt） 使用的时候 需要加载两部分： tokenizer、 model
# 每个模型的token划分方法 和 词表都不同

# modelscope 国内特色版本的huggingface
# pip install modelscope
# 下载命令 modelscope download --model google-bert/bert-base-chinese --local_dir models/google-bert/bert-base-chinese
# https://www.modelscope.cn/models/google-bert/bert-base-chinese/

# https://huggingface.co/google-bert/bert-base-chinese 谷歌官方推出原始的bert版本，用于中文的建模
tokenizer = AutoTokenizer.from_pretrained("../models/google-bert/bert-base-chinese")
model = AutoModelForMaskedLM.from_pretrained("../models/google-bert/bert-base-chinese") # 400mb

# 待处理的文本
text = "我喜欢人工智能westart鋐" # 分词 subword
# ['[CLS]', '我', '喜', '欢', '人', '工', '智', '能', '。', '[SEP]']
# bert 字作为token
# token： 文本切分之后的最小单位，输入到模型的单位
# Special Tokens: [CLS] [SEP] 标记特殊的用途的token
# [CLS]		BERT 句首标记，聚合句子表示用于分类
# [SEP]     句子分隔（单句结尾或句子对中间）

# 使用分词器对文本进行编码
encoded_input = tokenizer(text, return_tensors='pt') # 文本转换为token，token 编码
print("编码后的输入张量：")
print(encoded_input)
print("-----------------------")

"""
{'input_ids': tensor([[ 101,  782, 2339, 3255, 5543,  511,  102,  782, 2339, 3255, 5543,  511, 102]]), 
'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]]),  句子1 句子2
'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])} 有效输入
"""
# input_ids token 在 词表中的次序
# token_type_ids 区分 句子1的token  句子2的token
# attention_mask 告诉模型哪些token有效，哪些无效（padding）


# 打印分词结果
tokens = tokenizer.convert_ids_to_tokens(encoded_input['input_ids'][0])
print("分词结果：")
print(tokens)


# 增加一个文本示例
text_to_feature_extraction = "自然语言处理是人工智能的一个重要分支。"

# 对文本进行编码
encoded_input_features = tokenizer(text_to_feature_extraction, return_tensors='pt')

# 将编码后的输入传递给模型，禁用梯度计算以节省内存和提高速度
with torch.no_grad():
    # 语法糖，字典中的键值对作为关键字参数传递给模型
    outputs = model(**encoded_input_features, output_hidden_states=True)

# outputs是一个包含多个元素的元组或对象
# 第一个元素是 logits (用于完形填空任务)，第二个元素是隐藏层状态
# outputs.hidden_states 包含了所有层的隐藏层输出，是一个元组
# 最后一个元素是最后一层的输出，倒数第二个是倒数第二层的输出
last_hidden_state = outputs.hidden_states[-1]

# 也可以访问倒数第二层
second_to_last_hidden_state = outputs.hidden_states[-2]

print("文本的token ID数量:", encoded_input_features['input_ids'].shape[1])
print("最后一层隐藏层输出的形状:", last_hidden_state.shape)
print("倒数第二层隐藏层输出的形状:", second_to_last_hidden_state.shape)

# 隐藏层输出的形状为：[batch_size, sequence_length, hidden_size]
# last_hidden_state[0] 代表批次中的第一个（也是唯一一个）样本
# last_hidden_state[0][0] 代表 [CLS] token 的向量表示
cls_embedding = last_hidden_state[0][0]
print("[CLS] token 的向量表示形状:", cls_embedding.shape)

# 获取整个序列的特征向量（例如，取所有 token 向量的平均值）
# 这是一个简单的池化策略
mean_pooling_embedding = torch.mean(last_hidden_state[0], dim=0)
print("通过均值池化得到的序列特征向量形状:", mean_pooling_embedding.shape)

# 句子有n个token 输入， 提取特征 n * 768 维度