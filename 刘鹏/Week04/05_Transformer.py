import numpy as np  # 导入 numpy 库，用于科学计算
import torch  # 导入 torch 库，用于构建神经网络
import torch.nn as nn  # 导入 torch.nn 库，包含了各种神经网络层
from collections import Counter

# --- 全局参数 ---
d_k = 64  # Q 和 K 向量的维度
d_v = 64  # V 向量的维度
d_embedding = 128  # 词嵌入的维度
n_heads = 8  # 多头注意力机制中头的数量
n_layers = 6  # 编码器和解码器的层数
batch_size = 3  # 训练批次大小
epochs = 10  # 训练轮次


# --- 注意力机制模块 ---

class ScaledDotProductAttention(nn.Module):
    """
    缩放点积注意力模块
    根据 Q、K、V 计算注意力分数和上下文向量
    """

    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        # 1. 计算注意力分数
        # 将 Q 与 K 的转置相乘，并除以 sqrt(d_k) 进行缩放
        # scores 的维度: [batch_size, n_heads, len_q, len_k]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)

        # 2. 应用注意力掩码
        # 将 attn_mask 中为 True 的位置的 scores 替换为一个极小值 (-1e9)
        # 这样在 softmax 之后，这些位置的权重将接近于0
        scores.masked_fill_(attn_mask, -1e9)

        # 3. 对分数进行 softmax 归一化
        # 沿着最后一个维度 (len_k) 进行 softmax
        # weights 的维度: [batch_size, n_heads, len_q, len_k]
        weights = nn.Softmax(dim=-1)(scores)

        # 4. 计算上下文向量
        # 将归一化后的权重与 V 相乘
        # context 的维度: [batch_size, n_heads, len_q, dim_v]
        context = torch.matmul(weights, V)

        # 返回上下文向量和注意力权重
        return context, weights


# 层的用途是什么？
# 层的输入和输出维度是什么？
class MultiHeadAttention(nn.Module):
    """
    多头注意力机制模块
    将输入投影到多个子空间，并并行计算注意力，最后将结果拼接
    """

    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        # 线性投影层，用于生成 Q, K, V
        self.W_Q = nn.Linear(d_embedding, d_k * n_heads)
        self.W_K = nn.Linear(d_embedding, d_k * n_heads)
        self.W_V = nn.Linear(d_embedding, d_v * n_heads)
        # 最后的线性层，将拼接后的多头输出投影回原始维度
        self.linear = nn.Linear(n_heads * d_v, d_embedding)
        # 层归一化（Layer Normalization）， 样本的特征进行归一化，使其均值为 0，方差为 1
        self.layer_norm = nn.LayerNorm(d_embedding)

    def forward(self, Q, K, V, attn_mask):
        # Q, K, V 的维度: [batch_size, len_q/k/v, d_embedding]

        residual, batch_size = Q, Q.size(0)

        # 1. 线性投影并重塑
        # 将输入 Q, K, V 投影到多头子空间，并调整维度以便并行计算
        # .view() 和 .transpose() 操作将维度变为: [batch_size, n_heads, len_q/k/v, d_q=k/v]
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)

        # 2. 复制注意力掩码
        # 将 attn_mask 复制 n_heads 次，以适应多头注意力
        # attn_mask 的维度: [batch_size, n_heads, len_q, len_k]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        # 3. 计算缩放点积注意力
        context, weights = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        # context 的维度: [batch_size, n_heads, len_q, dim_v]
        # weights 的维度: [batch_size, n_heads, len_q, len_k]

        # 4. 拼接多头结果
        # 调整维度并使用 .contiguous() 确保内存连续，然后用 .view() 将多头结果拼接
        # context 的维度: [batch_size, len_q, n_heads * dim_v]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)

        # 5. 最终线性投影和层归一化
        # 将拼接后的结果通过线性层投影回 d_embedding 维度
        output = self.linear(context)
        # 将输出与残差连接相加，并进行层归一化
        # output 的维度: [batch_size, len_q, d_embedding]
        output = self.layer_norm(output + residual)

        # 返回最终输出和注意力权重
        return output, weights


# --- 前馈网络模块 ---

class PoswiseFeedForwardNet(nn.Module):
    """
    逐位置前馈网络模块
    对序列中的每个位置独立地应用一个全连接前馈网络
    """

    def __init__(self, d_ff=2048):
        super(PoswiseFeedForwardNet, self).__init__()
        # 两个一维卷积层，相当于两个全连接层 （实现的结果、做维度变换）
        self.conv1 = nn.Conv1d(in_channels=d_embedding, out_channels=d_ff, kernel_size=1)  # nn.Linear  nn.Conv1d 维度转换
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_embedding, kernel_size=1)
        # 层归一化
        self.layer_norm = nn.LayerNorm(d_embedding)

    def forward(self, inputs):
        # inputs 的维度: [batch_size, len_q, d_embedding]

        residual = inputs

        # 1. 维度转换并应用第一个卷积层
        # inputs.transpose(1, 2) 将维度变为 [batch_size, d_embedding, len_q]
        # 卷积操作后，output 的维度: [batch_size, d_ff, len_q]
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))

        # 2. 应用第二个卷积层并转换维度
        # 卷积操作后，output 的维度: [batch_size, d_embedding, len_q]
        # .transpose(1, 2) 将维度恢复为 [batch_size, len_q, d_embedding]
        output = self.conv2(output).transpose(1, 2)

        # 3. 残差连接和层归一化
        output = self.layer_norm(output + residual)

        return output


# --- 位置编码和掩码函数 ---

def get_sin_enc_table(n_position, embedding_dim):
    """
    生成正弦位置编码表
    用于在序列中引入词语的绝对位置信息
    """
    # 初始化正弦编码表
    sinusoid_table = np.zeros((n_position, embedding_dim))
    # 计算不同位置和维度的角度
    for pos_i in range(n_position):
        for hid_j in range(embedding_dim):
            angle = pos_i / np.power(10000, 2 * (hid_j // 2) / embedding_dim)
            sinusoid_table[pos_i, hid_j] = angle

    # 将偶数维应用 sin 函数，奇数维应用 cos 函数
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

    # 转换为 PyTorch 张量
    return torch.FloatTensor(sinusoid_table)


def get_attn_pad_mask(seq_q, seq_k):
    """
    生成填充注意力掩码
    用于在注意力计算中忽略填充 <pad> 词语 特殊的词
    """
    # seq_q 的维度: [batch_size, len_q]
    # seq_k 的维度: [batch_size, len_k]

    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()

    # 找到 seq_k 中所有值为 0 (<pad>) 的位置
    # pad_attn_mask 的维度: [batch_size, 1, len_k]
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)

    # 将掩码扩展到与注意力分数相同的形状
    # pad_attn_mask 的维度: [batch_size, len_q, len_k]
    pad_attn_mask = pad_attn_mask.expand(batch_size, len_q, len_k)

    return pad_attn_mask


def get_attn_subsequent_mask(seq):
    """
    生成后续注意力掩码 (仅用于解码器)
    用于在注意力计算中忽略当前位置之后的信息，防止信息泄露
    """
    # seq 的维度: [batch_size, seq_len]
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    # 创建一个上三角矩阵，k=1 表示主对角线上方的元素为 1
    # subsequent_mask 的维度: [batch_size, seq_len, seq_len]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    # 转换为 PyTorch 字节张量 (布尔类型)
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()

    return subsequent_mask


# --- 编码器和解码器模块 ---

class EncoderLayer(nn.Module):
    """
    编码器的一层
    包含一个多头自注意力层和一个位置前馈网络
    """

    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        # enc_inputs 的维度: [batch_size, seq_len, d_embedding]
        # enc_self_attn_mask 的维度: [batch_size, seq_len, seq_len]

        # 将相同的 Q, K, V 输入多头自注意力层
        enc_outputs, attn_weights = self.enc_self_attn(enc_inputs, enc_inputs,
                                                       enc_inputs, enc_self_attn_mask)

        # 将自注意力输出输入位置前馈网络
        enc_outputs = self.pos_ffn(enc_outputs)

        # 返回最终输出和注意力权重
        return enc_outputs, attn_weights


class Encoder(nn.Module):
    """
    Transformer 编码器
    由词嵌入层、位置嵌入层和多个编码器层组成
    """

    def __init__(self, corpus):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(len(corpus.src_vocab), d_embedding)
        # 从预计算的位置编码表初始化位置嵌入层，并冻结参数
        self.pos_emb = nn.Embedding.from_pretrained(
            get_sin_enc_table(corpus.src_len + 1, d_embedding), freeze=True)
        # 堆叠 n_layers 个编码器层
        self.layers = nn.ModuleList(EncoderLayer() for _ in range(n_layers))

    def forward(self, enc_inputs):
        # enc_inputs 的维度: [batch_size, source_len]

        # 生成位置索引序列
        pos_indices = torch.arange(1, enc_inputs.size(1) + 1).unsqueeze(0).to(enc_inputs)

        # 词嵌入和位置嵌入相加
        enc_outputs = self.src_emb(enc_inputs) + self.pos_emb(pos_indices)

        # 生成填充注意力掩码
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)

        enc_self_attn_weights = []
        # 逐层通过编码器层
        for layer in self.layers:
            enc_outputs, enc_self_attn_weight = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attn_weights.append(enc_self_attn_weight)

        # 返回最终输出和所有层的注意力权重
        return enc_outputs, enc_self_attn_weights


class DecoderLayer(nn.Module):
    """
    解码器的一层
    包含一个多头自注意力层、一个编码器-解码器注意力层和一个位置前馈网络
    """

    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        # dec_inputs 的维度: [batch_size, target_len, d_embedding]
        # enc_outputs 的维度: [batch_size, source_len, d_embedding]

        # 1. 第一个注意力子层: 多头自注意力
        # Q, K, V 都来自解码器输入
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs,
                                                        dec_inputs, dec_self_attn_mask)

        # 2. 第二个注意力子层: 编码器-解码器注意力
        # Q 来自解码器输出，K, V 来自编码器输出
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs,
                                                      enc_outputs, dec_enc_attn_mask)

        # 3. 位置前馈网络
        dec_outputs = self.pos_ffn(dec_outputs)

        # 返回最终输出和两个注意力层的权重
        return dec_outputs, dec_self_attn, dec_enc_attn


class Decoder(nn.Module):
    """
    Transformer 解码器
    由词嵌入层、位置嵌入层和多个解码器层组成
    """

    def __init__(self, corpus):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(len(corpus.tgt_vocab), d_embedding)
        # 从预计算的位置编码表初始化位置嵌入层，并冻结参数
        self.pos_emb = nn.Embedding.from_pretrained(
            get_sin_enc_table(corpus.tgt_len + 1, d_embedding), freeze=True)
        # 堆叠 n_layers 个解码器层
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        # dec_inputs 的维度: [batch_size, target_len]
        # enc_inputs 的维度: [batch_size, source_len]
        # enc_outputs 的维度: [batch_size, source_len, d_embedding]

        # 生成位置索引序列
        pos_indices = torch.arange(1, dec_inputs.size(1) + 1).unsqueeze(0).to(dec_inputs)

        # 词嵌入和位置嵌入相加
        dec_outputs = self.tgt_emb(dec_inputs) + self.pos_emb(pos_indices)

        # 1. 生成解码器自注意力掩码
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)
        # 将填充掩码和后续掩码相加，结果大于 0 的位置为 True
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask
                                       + dec_self_attn_subsequent_mask), 0)

        # 2. 生成编码器-解码器注意力掩码 (仅考虑填充)
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)

        dec_self_attns, dec_enc_attns = [], []
        # 逐层通过解码器层
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs,
                                                             dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)

        # 返回最终输出和所有层的注意力权重
        return dec_outputs, dec_self_attns, dec_enc_attns


# --- Transformer 模型 ---

class Transformer(nn.Module):
    """
    Transformer 模型的总框架
    由编码器、解码器和最后的线性投影层组成
    """

    def __init__(self, corpus):
        super(Transformer, self).__init__()
        self.encoder = Encoder(corpus)
        self.decoder = Decoder(corpus)
        # 最后的线性层，将解码器输出映射到目标词汇表大小
        self.projection = nn.Linear(d_embedding, len(corpus.tgt_vocab), bias=False)

    def forward(self, enc_inputs, dec_inputs):
        # enc_inputs 的维度: [batch_size, source_seq_len]
        # dec_inputs 的维度: [batch_size, target_seq_len]

        # 1. 编码器前向传播
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        # enc_outputs 的维度: [batch_size, source_len, d_embedding]

        # 2. 解码器前向传播
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        # dec_outputs 的维度: [batch_size, target_len, d_embedding]

        # 3. 线性投影
        dec_logits = self.projection(dec_outputs)
        # dec_logits 的维度: [batch_size, target_len, tgt_vocab_size]

        # 返回逻辑值和所有注意力权重
        return dec_logits, enc_self_attns, dec_self_attns, dec_enc_attns


class TranslationCorpus:
    """
    数据处理类，用于管理语料库、词汇表和批次生成
    """

    def __init__(self, sentences):
        self.sentences = sentences
        self.src_len = max(len(sentence[0].split()) for sentence in sentences) + 1
        self.tgt_len = max(len(sentence[1].split()) for sentence in sentences) + 2

        self.src_vocab, self.tgt_vocab = self.create_vocabularies()
        self.src_idx2word = {v: k for k, v in self.src_vocab.items()}
        self.tgt_idx2word = {v: k for k, v in self.tgt_vocab.items()}

    def create_vocabularies(self):
        src_counter = Counter(word for sentence in self.sentences for word in sentence[0].split())
        tgt_counter = Counter(word for sentence in self.sentences for word in sentence[1].split())

        src_vocab = {'<pad>': 0, **{word: i + 1 for i, word in enumerate(src_counter)}}
        tgt_vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2,
                     **{word: i + 3 for i, word in enumerate(tgt_counter)}}
        return src_vocab, tgt_vocab

    def make_batch(self, batch_size, test_batch=False):
        input_batch, output_batch, target_batch = [], [], []

        # 随机选择句子索引
        sentence_indices = torch.randperm(len(self.sentences))[:batch_size]

        for index in sentence_indices:
            src_sentence, tgt_sentence = self.sentences[index]

            # 将句子转换为索引序列
            src_seq = [self.src_vocab[word] for word in src_sentence.split()]
            tgt_seq = ([self.tgt_vocab['<sos>']] +
                       [self.tgt_vocab[word] for word in tgt_sentence.split()] +
                       [self.tgt_vocab['<eos>']])

            # 对序列进行填充
            src_seq += [self.src_vocab['<pad>']] * (self.src_len - len(src_seq))
            tgt_seq += [self.tgt_vocab['<pad>']] * (self.tgt_len - len(tgt_seq))

            input_batch.append(src_seq)
            # 在测试模式下，解码器输入只包含 <sos> 符号
            output_batch.append([self.tgt_vocab['<sos>']] +
                                ([self.tgt_vocab['<pad>']] * (self.tgt_len - 2)) if test_batch else tgt_seq[:-1])
            target_batch.append(tgt_seq[1:])

        return torch.LongTensor(input_batch), torch.LongTensor(output_batch), torch.LongTensor(target_batch)


# --- 训练和推理 ---

# 定义语料库
sentences = [
    # 原始句子 目标输出
    ['毛老师 喜欢 人工智能', 'TeacherMao likes AI'],  # 一个样本

    ['我 爱 学习 人工智能', 'I love studying AI'],
    ['深度学习 改变 世界', ' DL changed the world'],
    ['自然语言处理 很 强大', 'NLP is powerful'],
    ['神经网络 非常 复杂', 'Neural-networks are complex']
]

# 创建语料库实例
corpus = TranslationCorpus(sentences)

# 实例化模型、损失函数和优化器
model = Transformer(corpus)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# 模型训练
print("--- 开始训练 ---")
for epoch in range(epochs):
    optimizer.zero_grad()
    enc_inputs, dec_inputs, target_batch = corpus.make_batch(batch_size)
    outputs, _, _, _ = model(enc_inputs, dec_inputs)

    # 调整输出和目标张量维度以适应损失函数
    loss = criterion(outputs.view(-1, len(corpus.tgt_vocab)), target_batch.view(-1))

    if (epoch + 1) % 1 == 0:
        print(f"Epoch: {epoch + 1:04d} cost = {loss:.6f}")

    loss.backward()
    optimizer.step()

# 模型推理 (翻译)
print("\n--- 开始翻译 ---")
# 创建一个大小为 1 的测试批次
# enc_inputs 原始文本
# dec_inputs 目标文本，推理时候， 可以只包含 <sos>
enc_inputs, dec_inputs, target_batch = corpus.make_batch(batch_size=1, test_batch=True)

# 打印输入数据
print("编码器输入 :", enc_inputs)
print("解码器输入 :", dec_inputs)
print("目标数据 :", target_batch)

# 进行翻译预测
# 解码的结果
predict, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)

# 后处理预测结果
predict = predict.view(-1, len(corpus.tgt_vocab))
predict = predict.data.max(1, keepdim=True)[1]  # 贪婪解码（Greedy Decoding）

# 将索引转换为单词
translated_sentence = [corpus.tgt_idx2word[idx.item()] for idx in predict.squeeze()]
input_sentence = ' '.join([corpus.src_idx2word[idx.item()] for idx in enc_inputs[0]])

# 打印翻译结果
print(f"输入句子: '{input_sentence}'")
print(f"翻译结果: '{' '.join(translated_sentence)}'")

# Transformer 原理 和 内部结构 -》 面试、 理解专业名词使用用
# 1. 自己不会从头搭建 Transformer -》 用pytorch内置
# 2. 在特定的任务中会使用领域模型（BERT、GPT） 内置了 Transformer -》 用 BERT、GPT
