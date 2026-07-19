import torch # 深度学习框架
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

### 1. 创建 Tensor 张量 多维矩阵

# 直接从数据创建
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data) # 基础的数据结构
print("从数据创建的 Tensor:")
print(x_data)

# 从 NumPy 数组创建
import numpy as np
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print("\n从 NumPy 数组创建的 Tensor:")
print(x_np)

# 从其他 Tensor 创建
x_ones = torch.ones_like(x_data)  # 保持 x_data 的属性
print("\n从其他 Tensor 创建的 Tensor (ones_like):")
print(x_ones)

x_rand = torch.rand_like(x_data, dtype=torch.float)  # 覆盖 x_data 的数据类型
print("\n从其他 Tensor 创建的 Tensor (rand_like):")
print(x_rand)

# 创建特定大小的 Tensor
shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
print("\n创建特定大小的 Tensor:")
print(f"随机 Tensor: \n {rand_tensor}")
print(f"全1 Tensor: \n {ones_tensor}")
print(f"全0 Tensor: \n {zeros_tensor}")

# ---

### 2. Tensor 的属性

tensor = torch.rand(3, 4)
print("\n" + "="*50 + "\n")
print("Tensor 的属性:")
print(f"Shape: {tensor.shape}")      # 形状
print(f"数据类型: {tensor.dtype}")   # 数据类型
print(f"存储设备: {tensor.device}")   # 存储设备 (CPU 或 GPU)

# 将 Tensor 移动到 GPU
if torch.cuda.is_available():
    tensor = tensor.to("cuda")
    print(f"\nTensor 已移动到 GPU 设备: {tensor.device}")

# ---

### 3. Tensor 上的操作

# 基础数学运算
tensor_A = torch.tensor([[1, 2], [3, 4]])
tensor_B = torch.tensor([[5, 6], [7, 8]])

# 加法
tensor_C = tensor_A + tensor_B
print("\n" + "="*50 + "\n")
print("Tensor 的加法:")
print(f"A + B = \n{tensor_C}")

# 乘法 (逐元素乘法)
tensor_D = tensor_A * tensor_B
print("\nTensor 的逐元素乘法:")
print(f"A * B = \n{tensor_D}")

# 矩阵乘法
tensor_E = tensor_A @ tensor_B.T  # .T 是转置操作
print("\nTensor 的矩阵乘法:")
print(f"A @ B.T = \n{tensor_E}")

# 索引和切片
tensor_F = torch.arange(12).reshape(3, 4)
print("\nTensor 的索引和切片:")
print(f"原始 Tensor: \n{tensor_F}")
print(f"第一行: {tensor_F[0]}")
print(f"第二行第二列的元素: {tensor_F[1, 1]}")
print(f"所有行和第三列: \n{tensor_F[:, 2]}")

# 拼接
tensor_G = torch.cat([tensor_F, tensor_F], dim=1)  # 沿着维度1拼接
print("\nTensor 的拼接 (cat):")
print(tensor_G)

# 张量的 In-place 操作 (操作符带下划线)
# 例如，x.copy_(y), x.t_()
print("\n" + "="*50 + "\n")
print("Tensor 的 In-place 操作:")
print(f"原始 tensor_A: \n{tensor_A}")
tensor_A.add_(5)
print(f"使用 add_() 后: \n{tensor_A}")

# ---

### 4. Tensor 与 NumPy 的转换

# Tensor 转 NumPy
np_array_from_tensor = tensor_A.numpy()
print("\n" + "="*50 + "\n")
print("Tensor 转 NumPy:")
print(f"NumPy 数组: {np_array_from_tensor}")

# NumPy 转 Tensor
tensor_from_np = torch.from_numpy(np_array_from_tensor)
print("\nNumPy 转 Tensor:")
print(f"Tensor: \n{tensor_from_np}")

# 注意: Tensor 和 NumPy 数组会共享底层内存，修改一个会影响另一个。
np_array_from_tensor[0, 0] = 999
print(f"\n修改 NumPy 数组后，对应的 Tensor 也改变了: \n{tensor_from_np}")

