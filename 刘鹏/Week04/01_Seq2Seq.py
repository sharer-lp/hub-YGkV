import numpy as np
import torch
import random
import torch.nn as nn

# 构建语料库，每行包含中文、英文（解码器输入）和翻译成英文后的目标输出 3 个句子
sentences = [
    # sos：start of sentence， 模型翻译的起始位置。
    # eos：end of sentence，模型翻译的结束位置。

    # 原始文本使用 空格 划分为token

    # 原始文本、 解码输入、 解码输出
    ['毛老师 喜欢 人工智能', '<sos> TeacherMao likes AI', 'TeacherMao likes AI <eos>'],
    ['我 爱 学习 世界', '<sos> I love studying world', 'I love studying world <eos>'],
    ['深度学习 改变 世界', '<sos> DL changed the world', 'DL changed the world <eos>'],
    ['自然 语言 处理 很 强大', '<sos> NLP is so powerful', 'NLP is so powerful <eos>'],
    ['神经网络 非常 复杂', '<sos> Neural-Nets are complex', 'Neural-Nets are complex <eos>']
]

# 遍历每一个句子并将单词添加到词汇表中
word_list_cn, word_list_en = [], []
for s in sentences:
    word_list_cn.extend(s[0].split())
    word_list_en.extend(s[1].split())
    word_list_en.extend(s[2].split())

# 去重，得到没有重复单词的词汇表
word_list_cn = list(set(word_list_cn)) # 词典 包含所有的单词，唯一一次就够
word_list_en = list(set(word_list_en))

# 构建单词到索引的映射（embeeding、rnn、lstm）
# 构建词汇表
word2idx_cn = {w: i for i, w in enumerate(word_list_cn)}
word2idx_en = {w: i for i, w in enumerate(word_list_en)}

# 构建索引到单词的映射
idx2word_cn = {i: w for i, w in enumerate(word_list_cn)}
idx2word_en = {i: w for i, w in enumerate(word_list_en)}

# 计算词汇表的大小
voc_size_cn = len(word_list_cn)
voc_size_en = len(word_list_en)

# 定义一个函数，随机选择一个句子和词汇表生成输入、输出和目标数据
def make_data(sentences):
    # 随机选择一个句子进行训练
    random_sentence = random.choice(sentences)
    # 将输入句子中的单词转换为对应的索引
    encoder_input = np.array([[word2idx_cn[n] for n in random_sentence[0].split()]])
    # 将输出句子中的单词转换为对应的索引
    decoder_input = np.array([[word2idx_en[n] for n in random_sentence[1].split()]])
    # 将目标句子中的单词转换为对应的索引
    target = np.array([[word2idx_en[n] for n in random_sentence[2].split()]])
    # 将输入、输出和目标批次转换为 LongTensor
    encoder_input = torch.LongTensor(encoder_input)
    decoder_input = torch.LongTensor(decoder_input)
    target = torch.LongTensor(target)
    return encoder_input, decoder_input, target


# 使用 make_data 函数生成输入、输出和目标张量
encoder_input, decoder_input, target = make_data(sentences)
for s in sentences:  # 获取原始句子
    if all([word2idx_cn[w] in encoder_input[0] for w in s[0].split()]):
        original_sentence = s

print(" 原始句子：", original_sentence)  # 打印原始句子
print(" 编码器输入张量的形状：", encoder_input.shape)  # 打印输入张量形状
print(" 解码器输入张量的形状：", decoder_input.shape)  # 打印输出张量形状
print(" 目标张量的形状：", target.shape)  # 打印目标张量形状
print(" 编码器输入张量：", encoder_input)  # 打印输入张量
print(" 解码器输入张量：", decoder_input)  # 打印输出张量
print(" 目标张量：", target)  # 打印目标张量


# 定义编码器类，继承自 nn.Module
class Encoder(nn.Module):
    # 将输入的序列编码成一个包含其全部语义信息的上下文向量
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size  # 设置隐藏层大小
        self.embedding = nn.Embedding(input_size, hidden_size)  # 创建词嵌入层 input_size 词表大小
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)  # 创建 RNN 层

    def forward(self, inputs, hidden):  # 前向传播函数
        embedded = self.embedding(inputs)  # 将输入转换为嵌入向量
        output, hidden = self.rnn(embedded, hidden)  # 将嵌入向量输入 RNN 层并获取输出
        return output, hidden


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)

        # 全链接层，维度转换
        self.out = nn.Linear(hidden_size, output_size) # 全连接层， 目标词汇表的大小

    def forward(self, inputs, hidden):
        embedded = self.embedding(inputs)
        output, hidden = self.rnn(embedded, hidden)
        output = self.out(output)
        return output, hidden


n_hidden = 128  # 设置隐藏层数量
# 创建编码器和解码器
encoder = Encoder(voc_size_cn, n_hidden)
decoder = Decoder(n_hidden, voc_size_en)
print(' 编码器结构：', encoder)  # 打印编码器的结构
print(' 解码器结构：', decoder)  # 打印解码器的结构


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        # 初始化编码器和解码器
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_input, hidden, dec_input):  # 定义前向传播函数
        # enc_input 中文输入
        # hidden 初始化状态
        encoder_output, encoder_hidden = self.encoder(enc_input, hidden)

        # encoder_hidden 最终隐藏状态
        decoder_hidden = encoder_hidden

        # 英文输入序列
        decoder_output, _ = self.decoder(dec_input, decoder_hidden)
        return decoder_output


model = Seq2Seq(encoder, decoder)
print('模型结构：', model)


# 定义训练函数
def train_seq2seq(model, criterion, optimizer, epochs):
    for epoch in range(epochs): # epoch
        # 文本 -》 id
        encoder_input, decoder_input, target = make_data(sentences)

        # 初始化编码器的初始隐藏状态
        hidden = torch.zeros(1, encoder_input.size(0), n_hidden)

        optimizer.zero_grad()  # 梯度清零
        output = model(encoder_input, hidden, decoder_input)
        loss = criterion(output.view(-1, voc_size_en), target.view(-1)) # 分类

        if (epoch + 1) % 100 == 0:  # 打印损失
            print(f"Epoch: {epoch + 1:04d} cost = {loss:.6f}")
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数


# 训练模型
epochs = 500  # 训练轮次
criterion = nn.CrossEntropyLoss()  # 损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 优化器
train_seq2seq(model, criterion, optimizer, epochs)  # 调用函数训练模型


# 定义测试函数
def test_seq2seq(model, source_sentence):
    # 将输入的句子转换为索引
    encoder_input = np.array([[word2idx_cn[n] for n in source_sentence.split()]])

    # 构建输出的句子的索引，以 '<sos>' 开始，后面跟 '<eos>'，长度与输入句子相同
    decoder_input = np.array([word2idx_en['<sos>']] + [word2idx_en['<eos>']] * (len(encoder_input[0]) - 1))

    # 转换为 LongTensor 类型
    encoder_input = torch.LongTensor(encoder_input)
    decoder_input = torch.LongTensor(decoder_input).unsqueeze(0)

    hidden = torch.zeros(1, encoder_input.size(0), n_hidden)
    predict = model(encoder_input, hidden, decoder_input)
    predict = predict.data.max(2, keepdim=True)[1] # 概率最大的单词

    # 打印输入的句子和预测的句子
    print(source_sentence, '->', [idx2word_en[n.item()] for n in predict.squeeze()])


# 测试模型
test_seq2seq(model, '毛老师 喜欢 人工智能') # 没有解码器输入
