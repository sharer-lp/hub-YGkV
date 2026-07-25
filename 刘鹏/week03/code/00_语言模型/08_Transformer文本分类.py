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

# 建立标签-索引映射
label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]
index_to_label = {i: label for label, i in label_to_index.items()}

# 分词并建立词汇表
tokenized_texts = [list(jieba.cut(text)) for text in texts]
word_to_index = {'<pad>': 0, '<unk>': 1}
for tokens in tokenized_texts:
    for token in tokens:
        if token not in word_to_index:
            word_to_index[token] = len(word_to_index)
vocab_size = len(word_to_index)
max_len = 40

# 将文本和标签转换为张量
class TextDataset(Dataset):
    def __init__(self, tokenized_texts, labels, word_to_index, max_len):
        self.texts = tokenized_texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.word_to_index = word_to_index
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        indices = [self.word_to_index.get(word, 1) for word in self.texts[idx][:self.max_len]]
        indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]

text_dataset = TextDataset(tokenized_texts, numerical_labels, word_to_index, max_len)
dataloader = DataLoader(text_dataset, batch_size=32, shuffle=True)

# 2. 模型构建
# 位置编码模块
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# Transformer 分类器 ，没有解码器，文本分类只需要编码器部分
# Transformer block
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, dim_feedforward, output_dim, dropout=0.1):
        # vocab_size 词典的大小
        # d_model 潜入维度
        # nhead 头个数
        # num_encoder_layers Transformer编码器层数
        # dim_feedforward 中间层的维度
        # output_dim 输出的类别数量

        super(TransformerClassifier, self).__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        # 使用PyTorch内置Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)

        # 将多个编码器层堆叠起来
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # 分类器
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, src):
        # src 的形状: (batch_size, sequence_length)
        # torch.Size([32, 40])
        # 1. 词嵌入并进行缩放
        # torch.Size([32, 40, 32])
        src = self.embedding(src) * math.sqrt(self.d_model) # torch.Size([32, 40]) -> torch.Size([32, 40, 32])

        # 2. 位置编码
        src = self.pos_encoder(src) # torch.Size([32, 40, 32]) -> torch.Size([32, 40, 32])

        # 3. Transformer 编码器
        output = self.transformer_encoder(src) # torch.Size([32, 40, 32])

        # 4. 从序列中取第一个词（通常是 [CLS] 或用于分类的聚合词）
        # 这里我们简单地取序列所有位置特征的的平均值
        output = output.mean(dim=1) # torch.Size([32, 40, 32]) -> torch.Size([32, 32])

        # 5. 线性分类器
        output = self.fc(output) # torch.Size([32, 32]) -> torch.Size([32, 12])

        return output

# 模型参数设置
d_model = 32
nhead = 8
num_encoder_layers = 2
dim_feedforward = 128
output_dim = len(label_to_index)
dropout = 0.1

model = TransformerClassifier(vocab_size, d_model, nhead, num_encoder_layers, dim_feedforward, output_dim, dropout)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. 训练循环
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")

# 4. 模型评估与预测
def classify_text_transformer(text, model, word_to_index, max_len, index_to_label):
    model.eval()
    tokens = list(jieba.cut(text))
    indices = [word_to_index.get(word, 1) for word in tokens[:max_len]]
    indices += [0] * (max_len - len(indices))
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_index = torch.max(output, 1)
        predicted_index = predicted_index.item()
        predicted_label = index_to_label[predicted_index]

    return predicted_label

new_text = "帮我导航到北京"
predicted_class = classify_text_transformer(new_text, model, word_to_index, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text_transformer(new_text_2, model, word_to_index, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")