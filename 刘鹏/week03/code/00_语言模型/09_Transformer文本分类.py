import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import jieba

# 1. 数据预处理
# 加载数据
dataset = pd.read_csv("../Week01/dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

# 建立标签-索引映射，便于将文本标签转换为数字
label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]
index_to_label = {i: label for label, i in label_to_index.items()}

# 使用 jieba 对文本进行分词，并构建词汇表
tokenized_texts = [list(jieba.cut(text)) for text in texts]
# 初始化词汇表，<pad>用于填充，<unk>用于未知词
word_to_index = {'<pad>': 0, '<unk>': 1}
for tokens in tokenized_texts:
    for token in tokens:
        if token not in word_to_index:
            word_to_index[token] = len(word_to_index)
vocab_size = len(word_to_index)
max_len = 40


# 将文本和标签转换为 PyTorch 张量
class TextDataset(Dataset):
    def __init__(self, tokenized_texts, labels, word_to_index, max_len):
        self.texts = tokenized_texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.word_to_index = word_to_index
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # 将词语转换为索引，对于不在词汇表中的词使用 <unk> 的索引
        indices = [self.word_to_index.get(word, 1) for word in self.texts[idx][:self.max_len]]
        # 填充序列至最大长度
        indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]


# 创建数据集和数据加载器
text_dataset = TextDataset(tokenized_texts, numerical_labels, word_to_index, max_len)
dataloader = DataLoader(text_dataset, batch_size=32, shuffle=True)


# 2. 模型构建
# 位置编码模块：为序列中的每个词提供位置信息
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # 创建一个足够大的位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 计算不同维度的除数，使用对数空间以提高数值稳定性
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # 应用正弦函数到偶数维度，余弦函数到奇数维度
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        # 注册为模型缓冲区，它不会被训练
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 将位置编码加到输入中
        return x + self.pe[:, :x.size(1), :]


# Transformer 分类器，使用 PyTorch 内置的 nn.TransformerEncoder
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, output_dim, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        # 存储 d_model 以便在 forward 方法中使用，用于缩放嵌入向量
        self.d_model = d_model
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        # 位置编码层
        self.pos_encoder = PositionalEncoding(d_model)

        # 核心部分：使用 nn.TransformerEncoder 进行文本序列编码
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 分类器：一个简单的线性层
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, src):
        # src 的形状: (batch_size, sequence_length)
        # 1. 词嵌入并进行缩放（这是 Transformer 论文中的标准做法）
        src = self.embedding(src) * math.sqrt(self.d_model)
        # 2. 添加位置编码
        src = self.pos_encoder(src)

        # 3. 通过 Transformer 编码器层
        output = self.transformer_encoder(src)

        # 4. 从序列中聚合信息进行分类。这里我们简单地取所有词向量的平均值
        output = output.mean(dim=1)

        # 5. 线性分类器进行最终预测
        output = self.fc(output)
        return output


# 模型参数设置
d_model = 32
nhead = 8
num_layers = 2
dim_feedforward = 128
output_dim = len(label_to_index)
dropout = 0.1

# 实例化模型
model = TransformerClassifier(vocab_size, d_model, nhead, num_layers, dim_feedforward, output_dim, dropout)
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. 训练循环
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    running_loss = 0.0
    for inputs, labels in dataloader:
        optimizer.zero_grad()  # 梯度清零
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新模型参数
        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")


# 4. 模型评估与预测
def classify_text_transformer(text, model, word_to_index, max_len, index_to_label):
    model.eval()  # 设置模型为评估模式
    tokens = list(jieba.cut(text))
    # 将输入文本转换为模型可接受的索引张量
    indices = [word_to_index.get(word, 1) for word in tokens[:max_len]]
    indices += [0] * (max_len - len(indices))
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():  # 禁用梯度计算
        output = model(input_tensor)
        # 找出预测概率最高的类别索引
        _, predicted_index = torch.max(output, 1)
        predicted_index = predicted_index.item()
        # 将索引转换为对应的标签
        predicted_label = index_to_label[predicted_index]

    return predicted_label


# 示例预测
new_text = "帮我导航到北京"
predicted_class = classify_text_transformer(new_text, model, word_to_index, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text_transformer(new_text_2, model, word_to_index, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")