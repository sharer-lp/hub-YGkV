import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# BertForSequenceClassification bert 用于 文本分类
# Trainer： 直接实现 正向传播、损失计算、参数更新
# TrainingArguments： 超参数、实验设置

# 加载和预处理数据
dataset_df = pd.read_csv("../week03/code/01-intent-classify/assets/dataset/dataset.csv", sep="\t", header=None)

dataset_df = dataset_df.sample(frac=1, random_state=42).reset_index(drop=True)

# 初始化 LabelEncoder，用于将文本标签转换为数字标签
lbl = LabelEncoder()
# 拟合数据并转换前500个标签，得到数字标签
labels = lbl.fit_transform(dataset_df[1].values[:500])
# 提取前500个文本内容
texts = list(dataset_df[0].values[:500])

# 分割数据为训练集和测试集
x_train, x_test, train_labels, test_labels = train_test_split(texts,  # 文本数据
    labels,  # 对应的数字标签
    test_size=0.2,  # 测试集比例为20%
    stratify=labels  # 确保训练集和测试集的标签分布一致
)

# 从预训练模型加载分词器和模型
tokenizer = BertTokenizer.from_pretrained('../../models/google-bert/bert-base-chinese')
model = BertForSequenceClassification.from_pretrained('../../models/google-bert/bert-base-chinese', num_labels=len(lbl.classes_))
print(lbl.classes_)
print("类别数量：", len(lbl.classes_))

# 使用分词器对训练集和测试集的文本进行编码
# truncation=True：如果文本过长则截断
# padding=True：对齐所有序列长度，填充到最长
# max_length=64：最大序列长度
train_encodings = tokenizer(x_train, truncation=True, padding=True, max_length=64)
test_encodings = tokenizer(x_test, truncation=True, padding=True, max_length=64)

# 将编码后的数据和标签转换为 Hugging Face `datasets` 库的 Dataset 对象
train_dataset = Dataset.from_dict({'input_ids': train_encodings['input_ids'],  # 文本的token ID
    'attention_mask': train_encodings['attention_mask'],  # 注意力掩码
    'labels': train_labels  # 对应的标签
})
test_dataset = Dataset.from_dict(
    {'input_ids': test_encodings['input_ids'], 'attention_mask': test_encodings['attention_mask'],
        'labels': test_labels})


# 定义用于计算评估指标的函数
def compute_metrics(eval_pred):
    # eval_pred 是一个元组，包含模型预测的 logits 和真实的标签
    logits, labels = eval_pred
    # 找到 logits 中最大值的索引，即预测的类别
    predictions = np.argmax(logits, axis=-1)
    # 计算预测准确率并返回一个字典
    return {'accuracy': (predictions == labels).mean()}


# 配置训练参数
training_args = TrainingArguments(output_dir='./results',  # 训练输出目录，用于保存模型和状态
    num_train_epochs=4,  # 训练的总轮数
    per_device_train_batch_size=16,  # 训练时每个设备（GPU/CPU）的批次大小
    per_device_eval_batch_size=16,  # 评估时每个设备的批次大小
    warmup_steps=10,  # 学习率预热的步数，有助于稳定训练， step 定义为 一次 正向传播 + 参数更新
    learning_rate=2e-5,  # 学习率
    weight_decay=0.01,  # 权重衰减，用于防止过拟合
    logging_dir='./logs',  # 日志存储目录
    logging_steps=10,  # 每隔100步记录一次日志,25 步一个 epoch，10 步打一次日志，能看到训练 loss 变化
    eval_strategy="epoch",  # 每训练完一个 epoch 进行一次评估
    save_strategy="epoch",  # 每训练完一个 epoch 保存一次模型
    load_best_model_at_end=True,  # 训练结束后加载效果最好的模型
    metric_for_best_model="accuracy",  # 用于确定最佳模型的指标
    save_total_limit=1,  # 保存的模型数量上限
    report_to=[], # 不向wandb等平台上报日志，纯本地运行
    # max_steps=2,     # 用于确认整个链路通不通
    # report_to=[], # 用于确认整个链路通不通
)

# 实例化 Trainer 简化模型训练代码
trainer = Trainer(model=model,  # 要训练的模型
    args=training_args,  # 训练参数
    train_dataset=train_dataset,  # 训练数据集
    eval_dataset=test_dataset,  # 评估数据集
    compute_metrics=compute_metrics,  # 用于计算评估指标的函数
)

# 深度学习训练过程，数据获取，epoch batch 循环，梯度计算 + 参数更新

# 开始训练模型
trainer.train()
# 在测试集上进行最终评估
trainer.evaluate()

# trainer 是比较简单，适合训练过程比较规范化的模型
# 如果我要定制化训练过程，trainer无法满足
