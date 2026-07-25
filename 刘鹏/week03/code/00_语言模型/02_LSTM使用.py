import torch
import torch.nn as nn

# 设置参数
input_size = 10
hidden_size = 20
sequence_length = 1

# 1. 初始化 LSTM 模型
lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

# 2. 准备输入数据
x = torch.randn(1, sequence_length, input_size)

# 3. 初始化隐藏状态和细胞状态
h0 = torch.zeros(1, 1, hidden_size)
c0 = torch.zeros(1, 1, hidden_size)

# 4. 前向传播
# PyTorch LSTM 返回 output, (hn, cn)
output, (hn, cn) = lstm(x, (h0, c0))
print("PyTorch LSTM 模型的输出 (h_1):")
print(output)
print("PyTorch LSTM 模型的最终隐藏状态 (hn):")
print(hn)
print("PyTorch LSTM 模型的最终细胞状态 (cn):")
print(cn)
print("-" * 50)

# 5. 手动计算验证
# 获取模型参数（注意 PyTorch 将所有门的参数打包）
weight_ih = lstm.weight_ih_l0
weight_hh = lstm.weight_hh_l0
bias_ih = lstm.bias_ih_l0
bias_hh = lstm.bias_hh_l0

# 拆分参数
W_ii, W_if, W_ig, W_io = torch.chunk(weight_ih, 4, 0)
W_hi, W_hf, W_hg, W_ho = torch.chunk(weight_hh, 4, 0)
b_ii, b_if, b_ig, b_io = torch.chunk(bias_ih, 4, 0)
b_hi, b_hf, b_hg, b_ho = torch.chunk(bias_hh, 4, 0)

# 手动计算
x1_squeeze = x.squeeze(1)
h0_squeeze = h0.squeeze(0)
c0_squeeze = c0.squeeze(0)

# 遗忘门
f_t = torch.sigmoid(torch.matmul(x1_squeeze, W_if.t()) + b_if + torch.matmul(h0_squeeze, W_hf.t()) + b_hf)
# 输入门和候选细胞状态
i_t = torch.sigmoid(torch.matmul(x1_squeeze, W_ii.t()) + b_ii + torch.matmul(h0_squeeze, W_hi.t()) + b_hi)
g_t = torch.tanh(torch.matmul(x1_squeeze, W_ig.t()) + b_ig + torch.matmul(h0_squeeze, W_hg.t()) + b_hg)
# 更新细胞状态
c1_manual = f_t * c0_squeeze + i_t * g_t
# 输出门
o_t = torch.sigmoid(torch.matmul(x1_squeeze, W_io.t()) + b_io + torch.matmul(h0_squeeze, W_ho.t()) + b_ho)
# 更新隐藏状态
h1_manual = o_t * torch.tanh(c1_manual)

print("手动计算的 LSTM 隐藏状态 (h_1):")
print(h1_manual)
print("手动计算的 LSTM 细胞状态 (c_1):")
print(c1_manual)
print("-" * 50)

# 验证结果是否一致
print("手动计算与 PyTorch 模型输出是否接近：")
print(torch.allclose(output, h1_manual))
print(torch.allclose(cn, c1_manual))