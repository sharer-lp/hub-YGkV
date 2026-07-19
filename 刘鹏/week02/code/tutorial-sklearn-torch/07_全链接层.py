import torch

batch_size = 4 # 一次处理的样本数量 皮大小
in_features = 10 # 输入特征维度10
out_features = 5 # 输出特征维度5

input_tensor = torch.randn(batch_size, in_features) # 4个样本，每个样本10维度特征
manual_weight = torch.randn(out_features, in_features) # 5 个神经元，10维度的权重
manual_bias = torch.randn(out_features)

print("输入张量的形状:", input_tensor.shape)
print("权重矩阵的形状:", manual_weight.shape)
print("偏置向量的形状:", manual_bias.shape)

# --- 手动实现 ---
# 矩阵乘法：torch.matmul(input, weight.T)
# (batch_size, in_features) x (in_features, out_features)
manual_output = torch.matmul(input_tensor, manual_weight.T) + manual_bias

print("\n手动实现的输出形状:", manual_output.shape)
print("手动实现的输出:\n", manual_output)

# --- 使用 nn.Linear 验证 ---
# 1. 创建 nn.Linear 实例
linear_layer = torch.nn.Linear(in_features, out_features) # 全链接层（线性层），输入维度 -》 输出维度

# 2. 将手动创建的权重和偏置赋值给 nn.Linear 实例
# 注意：nn.Linear 的权重形状是 (out_features, in_features)
# 其 bias 的形状是 (out_features,)
linear_layer.weight.data = manual_weight
linear_layer.bias.data = manual_bias

# 3. 使用 nn.Linear 进行前向传播
linear_output = linear_layer(input_tensor)

print("\nnn.Linear 模块的输出形状:", linear_output.shape)
print("nn.Linear 模块的输出:\n", linear_output)

is_equal = torch.allclose(manual_output, linear_output)
print("\n手动实现和 nn.Linear 的结果是否相同:", is_equal)