import numpy as np
from sklearn.linear_model import LinearRegression

# 1. 生成模拟数据 (与之前相同)
X_numpy = np.random.rand(100, 1) * 10

# 人工模拟加入噪音
# 真实数据集
y_numpy = 2 * X_numpy + 1 + np.random.randn(100, 1) # y = 2 * x + 1

# 2. 构建线性回归模型
model = LinearRegression() # 机器学习模型， y = ax + b
model.fit(X_numpy, y_numpy)

# 模型的 斜率  偏置
print(model.coef_, model.intercept_)