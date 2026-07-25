import torch
import torch.nn as nn # 深度学习的搭建

# 设置参数
input_size = 10
hidden_size = 20
sequence_length = 1

# 初始化 RNN 模型
# input_size 每个时间步输入向量的维度（如词嵌入维度）
# hidden_size 隐藏状态的维度，决定RNN的"记忆容量"
# batch_first 输入格式
    # True: 输入形状为 (batch_size, seq_len, input_size)
    # False（默认）: 输入形状为 (seq_len, batch_size, input_size)
rnn = nn.RNN(input_size, hidden_size, batch_first=True)

# 准备输入数据
# batch_size = 1, sequence_length = 10, input_size = 10
x = torch.randn(1, sequence_length, input_size) # 一个样本，样本长度是1，每一步输入的特征维度

# 初始化隐藏状态，手动初始化，自己随机初始化
h0 = torch.zeros(1, 1, hidden_size)

# 前向传播
# PyTorch 的 RNN 会自动处理所有时间步
output, hn = rnn(x, h0) # h0 -》 h1 -》 h2
print("PyTorch RNN 模型的输出 (h_1):")
print(output.shape) # 输入 1 * 3 * 10 -》 1 * 3 * 20
print("PyTorch RNN 模型的最终隐藏状态 (hn):")
print(hn.shape)
print("-" * 50)

# 5. 手动计算验证
# 获取模型参数
W_ih = rnn.weight_ih_l0
W_hh = rnn.weight_hh_l0
b_ih = rnn.bias_ih_l0
b_hh = rnn.bias_hh_l0

# 理论计算
# h_1 = tanh(W_ih * x_1 + b_ih + W_hh * h_0 + b_hh)
# PyTorch 默认输入是 (seq_len, batch, input_size)，这里我们使用了 batch_first=True，所以输入是 (batch, seq_len, input_size)
x1_squeeze = x.squeeze(1) # 移除 sequence_length 维度，变为 (1, 10)
h0_squeeze = h0.squeeze(0) # 移除 num_layers 维度，变为 (1, 20)
h1_manual = torch.tanh(torch.matmul(x1_squeeze, W_ih.t()) + b_ih + torch.matmul(h0_squeeze, W_hh.t()) + b_hh)

print("手动计算的 RNN 隐藏状态 (h_1):")
print(h1_manual)
print("-" * 50)

# 验证结果是否一致
print("手动计算与 PyTorch 模型输出是否接近：")
print(torch.allclose(output, h1_manual))