import torch
import torch.nn as nn

# 全连接网络 （前馈神经网络）

# 使用 nn.Sequential 直接构建一个模型对象
# 不需要单独创建一个类
# 输入层10 -》 隐藏层20 -》 隐藏层30 -》 输出层5
mlp_model = nn.Sequential(
    # 输入层到第一个隐藏层
    # 输入是X 维度是10， 维度转换到20 （20个神经元）
    nn.Linear(in_features=10, out_features=20),
    nn.Sigmoid(),

    # 第一个隐藏层到第二个隐藏层
    # 输入的X 维度20， 维度转换到30 （30个神经元）
    nn.Linear(in_features=20, out_features=30),
    nn.ReLU(),

    # 第二个隐藏层到输出层
    # 输入的X 维度30， 维度转换到5 （5个神经元）
    nn.Linear(in_features=30, out_features=5)
)
print("模型结构:\n", mlp_model)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        super(MLP, self).__init__()

        # 使用 nn.Sequential 封装多层网络
        # 这是一种简洁且常用的方式，可以方便地组织和查看网络结构
        self.network = nn.Sequential(
            # 第1层：从 input_size 到 hidden_size1
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(), # 增加模型的复杂度，非线性

            # 第2层：从 hidden_size1 到 hidden_size2
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),

            # 第3层：从 hidden_size2 到 hidden_size3
            nn.Linear(hidden_size2, hidden_size3),
            nn.ReLU(),

            # 输出层：从 hidden_size3 到 output_size
            nn.Linear(hidden_size3, output_size)
        )

    def forward(self, x):
        return self.network(x)


# --- 模型参数和实例化 ---
input_size = 10
hidden_size1 = 20
hidden_size2 = 30
hidden_size3 = 40
output_size = 5

# 实例化模型
model = MLP(input_size, hidden_size1, hidden_size2, hidden_size3, output_size)
print("模型结构:\n", model)

batch_size = 64
dummy_input = torch.randn(batch_size, input_size)

# 执行前向传播
output = model(dummy_input)

print("\n输入张量的形状:", dummy_input.shape)
print("输出张量的形状:", output.shape)