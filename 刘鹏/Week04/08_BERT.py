import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from collections import Counter
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import tqdm
import random
import pickle
import tqdm
import math

class BertEmbedding(nn.Module):
    '''
    BertEmbedding包括三部分, 三部分相加并输出:
    1. TokenEmbedding  /  2. PositionalEmbedding  /  3. SegmentEmbedding
    '''

    def __init__(self, vocab_size, embed_size, dropout=0.1):
        super(BertEmbedding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 词向量层
        self.token_embed = TokenEmbedding(vocab_size, embed_size)
        # 位置编码层
        self.position_embed = PositionalEmbedding(embed_size)
        # 句子类型编码层
        self.segment_embed = SegmentEmbedding(embed_size)

    def forward(self, sequence, segment_label):
        # 词嵌入、位置嵌入和句子嵌入相加
        x = self.token_embed(sequence) + self.position_embed(sequence) + self.segment_embed(segment_label)
        return self.dropout(x)


class TokenEmbedding(nn.Embedding):
    # 词嵌入层
    def __init__(self, vocab_size, embed_size=512):
        super().__init__(vocab_size, embed_size, padding_idx=0)


class PositionalEmbedding(nn.Module):
    # 位置编码层，使用正弦和余弦函数
    def __init__(self, embed_size, max_len=512):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, embed_size)  # (max_len, model_dim)
        pe.requires_grad = False  # 位置编码不需要梯度
        position = torch.arange(0, max_len).unsqueeze(1)  # (max_len, 1)
        # 计算分母，用于控制正弦余弦的频率
        div = torch.exp(torch.arange(0., embed_size, 2) * (- math.log(10000.) / embed_size))

        pe[:, 0::2] = torch.sin(position * div)  # 偶数位置使用sin
        pe[:, 1::2] = torch.cos(position * div)  # 奇数位置使用cos
        pe = pe.unsqueeze(0)  # (1, max_len, model_dim)
        self.register_buffer('pe', pe)  # 将位置编码存入缓冲区，不作为模型参数

    def forward(self, x):
        # 返回对应输入序列长度的位置编码
        # x.size(1) 获取输入序列的长度
        return self.pe[:, x.size(1)]  # (b, max_len, model_dim)


class SegmentEmbedding(nn.Embedding):
    # 句子类型嵌入层，用于区分句子A和句子B
    def __init__(self, embed_size=512):
        super().__init__(3, embed_size, padding_idx=0)  # 0: padding, 1: 句子A, 2: 句子B


# --------------------------------- TransformerBlock -------------------------------------------

class TransformerBlock(nn.Module):
    """
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, head, feed_forward_hidden, dropout):
        super(TransformerBlock, self).__init__()
        # 多头注意力机制
        self.attention = MultiHeadedAttention(hidden, head, dropout=dropout)
        # 前馈网络
        self.feed_forward = FeedForward(hidden, feed_forward_hidden, dropout=dropout)
        # 自注意力层的残差连接和层归一化
        self.attn_sublayer = SubLayerConnection(hidden, dropout)
        # 前馈网络的残差连接和层归一化
        self.ff_sublayer = SubLayerConnection(hidden, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # 自注意力模块的前向传播
        x = self.attn_sublayer(x, lambda x: self.attention(x, x, x, mask))
        # 前馈网络的前向传播
        x = self.ff_sublayer(x, self.feed_forward)
        return self.dropout(x)


class LayerNorm(nn.Module):
    # 层归一化
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # 可学习的缩放参数
        self.alpha = nn.Parameter(torch.ones(features))
        # 可学习的平移参数
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        x_mean = x.mean(-1, keepdim=True)
        x_std = x.std(-1, keepdim=True)
        return self.alpha * (x - x_mean) / (x_std + self.eps) + self.beta


class SubLayerConnection(nn.Module):
    # 残差连接和层归一化
    def __init__(self, hidden, dropout):
        super(SubLayerConnection, self).__init__()
        self.layer_norm = LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # 归一化后执行子层操作，然后进行残差连接和dropout
        return x + self.dropout(sublayer(self.layer_norm(x)))


def attention(q, k, v, mask=None, dropout=None):
    # q=k=v: (b, head, max_len, dk)
    # mask: (b, max_len, max_len)
    dk = q.size(-1)
    # 计算注意力分数，q和k的乘积除以根号dk
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dk)  # (b, head, max_len, max_len)

    if mask is not None:
        # 对padding部分进行掩码，使用一个很大的负值
        scores = scores.masked_fill(mask == 0, -1e9)  # padding_mask, 极小值填充

    # softmax操作得到注意力权重
    attention = F.softmax(scores, dim=-1)  # (b, head, max_len, max_len)
    if dropout is not None:
        attention = dropout(attention)  # (b, head, max_len, max_len)

    # 注意力权重和v相乘
    return torch.matmul(attention, v), attention


class MultiHeadedAttention(nn.Module):
    # 多头注意力机制
    def __init__(self, hidden, head, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        self.dk = hidden // head  # 每个头的维度
        self.head = head  # 头数
        # 3个线性层分别用于q, k, v的投影
        self.input_linears = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(3)])
        # 输出线性层
        self.output_linear = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)
        self.attn = None

    def forward(self, q, k, v, mask=None):
        # q=k=v: (b, max_len, hidden)
        batch_size = q.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1)  # (b, 1, max_len, max_len)

        # 线性投影并进行分头操作
        # q,k,v: (b, head, max_len, dk)
        q, k, v = [linear(x).view(batch_size, -1, self.head, self.dk).transpose(1, 2)
                   for linear, x in zip(self.input_linears, (q, k, v))]

        # 调用自定义的 attention 函数
        # x: (b, head, max_len, dk)
        # attn: (b, head, max_len, max_len)
        x, self.attn = attention(q, k, v, mask=mask, dropout=self.dropout)
        # 将多头的结果拼接，并进行线性投影
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.dk)  # (b, max_len, hidden)
        return self.output_linear(x)  # (b, max_len, hidden)


class FeedForward(nn.Module):
    # 前馈网络
    def __init__(self, hidden, ff_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(hidden, ff_dim)
        self.linear2 = nn.Linear(ff_dim, hidden)
        self.dropout = nn.Dropout(dropout)
        self.activation = GLUE()  # 使用GLUE激活函数

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class GLUE(nn.Module):
    # GLUE激活函数
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


# ------------------------------------ Bert ----------------------------------

class Bert(nn.Module):
    '''
    BertEmbedding + TransformerBlock
    '''

    def __init__(self, vocab_size, hidden=768, n_layers=12, head=12, dropout=0.1):
        super(Bert, self).__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.head = head
        self.feed_forward_hidden = hidden * 4
        # 嵌入层
        self.embedding = BertEmbedding(vocab_size=vocab_size, embed_size=hidden, dropout=dropout)
        # 堆叠多个Transformer Block
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, head, hidden * 4, dropout) for _ in range(n_layers)])

    def forward(self, x, segment_info):
        # x: (b, max_len)
        # segment_info: (b, max_len)
        # 创建注意力掩码，将padding部分设置为0
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1)  # (b, max_len, max_len)
        # 嵌入层的前向传播
        x = self.embedding(x, segment_info)  # (b, max_len, embed_size)
        # 循环执行Transformer Block
        for transformer in self.transformer_blocks:
            x = transformer(x, mask)
        return x


# ---------------------------------- BertDataset ------------------------------------

class BertDataset(Dataset):
    def __init__(self, corpus_path, vocab, seq_len, encoding='utf-8', corpus_lines=None):
        self.vocab = vocab
        self.seq_len = seq_len

        # 加载语料库
        with open(corpus_path, encoding=encoding) as f:
            self.datas = [line[:-1].split("\t")
                          for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines)]

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        # 随机选择下一句
        t1, (t2, is_next_label) = self.datas[item][0], self.random_sent(item)
        # 对句子进行随机掩码
        t1_random, t1_label = self.random_word(t1)
        t2_random, t2_label = self.random_word(t2)

        # 添加特殊标记 [CLS] 和 [SEP]
        t1 = [self.vocab.sos_index] + t1_random + [self.vocab.eos_index]
        t2 = t2_random + [self.vocab.eos_index]

        t1_label = [self.vocab.pad_index] + t1_label + [self.vocab.pad_index]
        t2_label = t2_label + [self.vocab.pad_index]

        # 拼接句子
        bert_input = (t1 + t2)[:self.seq_len]
        bert_label = (t1_label + t2_label)[:self.seq_len]
        # 构造句子类型标签
        segment_label = ([1 for _ in range(len(t1))] + [2 for _ in range(len(t2))])[:self.seq_len]

        # 如果句子不足进行padding
        padding = [self.vocab.pad_index for _ in range(self.seq_len - len(bert_input))]
        bert_input.extend(padding)
        bert_label.extend(padding)
        segment_label.extend(padding)

        output = {
            'bert_input': bert_input,
            'bert_label': bert_label,
            'segment_label': segment_label,
            'is_next': is_next_label
        }
        return {key: torch.tensor(value) for key, value in output.items()}

    def random_word(self, sentence):
        tokens = sentence.split()
        sent_label = []  # 0是没有被mask，有数值的是mask的位置

        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15
                # 80%用mask替换
                if prob < 0.8:
                    tokens[i] = self.vocab.mask_index
                # 10%词表中随机一个词替换
                elif prob < 0.9:
                    tokens[i] = random.randrange(len(self.vocab))
                else:
                    # 10%用当前词不变
                    tokens[i] = self.vocab.stoi.get(tokens, self.vocab.unk_index)
                sent_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))
            else:
                tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                sent_label.append(0)

        return tokens, sent_label

    def random_sent(self, item):
        # 50%概率选择下一句，50%概率选择随机一句
        if random.random > 0.5:
            return self.datas[item][1], 1  # 是下一句
        else:
            return self.datas[random.randrange(len(self.datas))][1], 0  # 不是下一句


# --------------------------------------- Bert预训练任务及BertTrainer ---------------------------------

class BertLM(nn.Module):
    '''
    BERT 语言模型 (两个预训练任务)
    Masked Language Model + Next Sentence Prediction Model
    '''

    def __init__(self, bert, vocab_size):
        super(BertLM, self).__init__()
        self.bert = bert
        # 掩码语言模型，用于预测被掩码的词
        self.mask_lm = MaskedLanguageModel(self.bert.hidden, vocab_size)
        # 下一句预测模型
        self.next_sentence = NextSentencePrediction(self.bert.hidden)

    def forward(self, x, segment_label):
        out = self.bert(x, segment_label)  # (b, max_len, hidden)
        return self.mask_lm(out), self.next_sentence(out)  # (b, max_len, vocab_size) / (b, 2)


class MaskedLanguageModel(nn.Module):
    '''
    n分类问题：n-class = vocab_size
    '''

    def __init__(self, hidden, vocab_size):
        super(MaskedLanguageModel, self).__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))  # 对所有位置进行预测


class NextSentencePrediction(nn.Module):
    """
    2-class分类: is_next, is_not_next
    """

    def __init__(self, hidden):
        super(NextSentencePrediction, self).__init__()
        self.linear = nn.Linear(hidden, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        # 只在 [CLS] token 的输出上进行预测 (即序列的第一个位置)
        return self.softmax(self.linear(x[:, 0]))  # 只在x的0位置上进行预测


class BertTtrainer:
    '''
    Bert预训练模型包括两个LM预训练任务:
    1. Masked Language Model (掩码语言模型)
    2. Next Sentence prediction (下一句预测)
    '''

    def __init__(self, bert, vocab_size, train_dataloader, test_dataloader=None, lr=1e-4,
                 betas=(0.9, 0.999), weight_decay=0.01):
        self.bert = bert
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 将 BERT 模型和两个预训练任务组合
        self.bert_lm = BertLM(bert, vocab_size).to(self.device)

        if torch.cuda.device_count() > 1:
            # 如果有多个GPU，使用DataParallel
            self.bert_lm = nn.DataParallel(self.bert_lm)

        self.train_data = train_dataloader
        self.test_data = test_dataloader
        # 优化器
        self.optim = Adam(self.bert_lm.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        # 损失函数，忽略padding位置
        self.criterion = nn.NLLLoss(ignore_index=0)
        # 打印总参数数量
        print('Total Parameters:', sum([p.nelement() for p in self.bert_lm.parameters()]))

    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, mode='test')

    def iteration(self, epoch, data_loader, mode='train'):
        total_loss = 0.0
        total_correct = 0
        total_element = 0

        for i, data in enumerate(data_loader):
            # 将数据移动到指定设备
            data = {key: value.to(self.device) for key, value in data.items()}
            # 前向传播
            mask_lm_output, next_sentence_output = self.bert_lm(data['bert_input'], data['segment_label'])
            # 计算掩码语言模型的损失
            mask_loss = self.criterion(mask_lm_output.transpose(1, 2), data['bert_label'])
            # 计算下一句预测的损失
            next_loss = self.criterion(next_sentence_output, data['is_next'])

            # 总损失为两个任务的损失之和
            loss = mask_loss + next_loss
            if mode == 'train':
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            # 计算下一句预测的准确率
            correct = next_sentence_output.argmax(dim=-1).eq(data['is_next']).sum().item()
            total_loss += loss.item()
            total_correct += correct
            total_element += data['is_next'].nelement()

        print('mode: %s, epoch:%d, avg_loss: %.5f, total_acc: %.5f' % (
            mode, epoch, total_loss / len(data_loader), total_correct * 100.0 / total_element))

    def save(self, epoch, save_path='bert_pretrain.model'):
        # 保存模型，先将模型移到CPU
        torch.save(self.bert.cpu(), save_path)
        self.bert.to(self.device)


model = Bert(3000, 768, 12, 12, 0.1)
print(model)