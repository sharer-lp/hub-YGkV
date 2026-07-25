import torch
import torch.nn as nn

# 设置参数
input_size = 10
hidden_size = 20
sequence_length = 1

# 1. 初始化 GRU 模型
gru = nn.GRU(input_size, hidden_size, batch_first=True)

# 2. 准备输入数据
x = torch.randn(1, sequence_length, input_size)

# 3. 初始化隐藏状态
h0 = torch.zeros(1, 1, hidden_size)

# 4. 前向传播
output, hn = gru(x, h0)
print("PyTorch GRU 模型的输出 (h_1):")
print(output)
print("PyTorch GRU 模型的最终隐藏状态 (hn):")
print(hn)
print("-" * 50)

# 5. 手动计算验证
# 获取模型参数
weight_ih = gru.weight_ih_l0
weight_hh = gru.weight_hh_l0
bias_ih = gru.bias_ih_l0
bias_hh = gru.bias_hh_l0

# 拆分参数
W_ir, W_iz, W_in = torch.chunk(weight_ih, 3, 0)
W_hr, W_hz, W_hn = torch.chunk(weight_hh, 3, 0)
b_ir, b_iz, b_in = torch.chunk(bias_ih, 3, 0)
b_hr, b_hz, b_hn = torch.chunk(bias_hh, 3, 0)

# 手动计算
x1_squeeze = x.squeeze(1)
h0_squeeze = h0.squeeze(0)

# 重置门
r_t = torch.sigmoid(torch.matmul(x1_squeeze, W_ir.t()) + b_ir + torch.matmul(h0_squeeze, W_hr.t()) + b_hr)
# 更新门
z_t = torch.sigmoid(torch.matmul(x1_squeeze, W_iz.t()) + b_iz + torch.matmul(h0_squeeze, W_hz.t()) + b_hz)
# 候选隐藏状态
h_cand = torch.tanh(torch.matmul(x1_squeeze, W_in.t()) + b_in + r_t * (torch.matmul(h0_squeeze, W_hn.t()) + b_hn))
# 更新隐藏状态
h1_manual = (1 - z_t) * h_cand + z_t * h0_squeeze

print("手动计算的 GRU 隐藏状态 (h_1):")
print(h1_manual)
print("-" * 50)

# 验证结果是否一致
print("手动计算与 PyTorch 模型输出是否接近：")
print(torch.allclose(output, h1_manual))