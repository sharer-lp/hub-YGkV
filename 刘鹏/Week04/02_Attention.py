# ===== 多头注意力机制（Multi-Head Attention）完整实现 =====
# 本文件实现了 Transformer 中的多头注意力机制。
# 整体流程：输入 → Q/K/V投影 → 多头拆分 → 缩放点积注意力 → 多头合并 → 输出投影
# 涉及知识：[[多头注意力机制]]、[[缩放点积注意力]]、[[PyTorch张量变换]]

import torch
import torch.nn.functional as F

# ===== 第 1 段：输入与配置 =====
# 做什么：创建输入数据，定义多头注意力的参数
# 形状：x = (batch_size=2, seq_len=3, feature_dim=4)
#   - 2个样本（batch），每个样本3个词（seq），每个词用4维向量表示（feature）

x = torch.randn(2, 3, 4)  # 创建输入张量: 形状 (2, 3, 4)

# 多头配置
num_heads = 2   # 注意力头数：把特征分成2组，每组独立做注意力
head_dim = 2    # 每个头的维度：feature_dim(4) / num_heads(2) = 2

# 约束检查：特征维度必须能被头数整除，否则无法均分
assert x.size(-1) == num_heads * head_dim  # 4 == 2*2 ✅


# ===== 第 2 段：线性投影生成 Q、K、V =====
# 做什么：用三个独立的线性层，把同一个输入 x 映射到三个不同的空间
# 为什么：Q/K/V 扮演不同角色（"找什么""被什么匹配""传递什么"），需要不同的权重
# 形状变化：(2, 3, 4) -> Linear -> (2, 3, 4)（形状不变，值被变换了）
# 涉及知识：[[人工神经元与基础组件]]（线性层）、[[缩放点积注意力]]（Q/K/V的角色）

linear_q = torch.nn.Linear(4, 4)  # Wq: 把输入投影到 Query 空间
linear_k = torch.nn.Linear(4, 4)  # Wk: 把输入投影到 Key 空间
linear_v = torch.nn.Linear(4, 4)  # Wv: 把输入投影到 Value 空间

Q = linear_q(x)  # Query: "我想找什么信息" — 形状 (2, 3, 4)
K = linear_k(x)  # Key:   "我能提供什么信息" — 形状 (2, 3, 4)
V = linear_v(x)  # Value: "我的实际内容是什么" — 形状 (2, 3, 4)


# ===== 第 3 段：多头拆分（split_heads） =====
# 做什么：把特征维度 F 拆成 (H个头, 每头D维)，让每个头独立处理一部分特征
# 为什么：单头只能学一种关注模式；多头让模型同时关注语法、语义等不同关系
# 形状变化：(B, S, F=4) -> view -> (B, S, H=2, D=2) -> transpose -> (B, H=2, S=3, D=2)
# 涉及知识：[[多头注意力机制]]（split详解）、[[PyTorch张量变换]]（view + transpose）

def split_heads(tensor, num_heads):
    """
    将特征维度拆分为多头格式。
    输入: (batch, seq_len, feature_dim)
    输出: (batch, num_heads, seq_len, head_dim)
    """
    batch_size, seq_len, feature_dim = tensor.size()
    head_dim = feature_dim // num_heads  # 每头的维度 = 总维度 / 头数

    # 第一步 view: (B, S, F) -> (B, S, H, D)
    #   把 4 维特征重新解释为 2组 x 2维
    # 第二步 transpose(1, 2): (B, S, H, D) -> (B, H, S, D)
    #   把"头"维度提到前面，这样每个头可以独立做矩阵乘法
    output = tensor.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    return output


Q = split_heads(Q, num_heads)  # (2, 2, 3, 2) — batch=2, heads=2, seq=3, head_dim=2
K = split_heads(K, num_heads)  # (2, 2, 3, 2)
V = split_heads(V, num_heads)  # (2, 2, 3, 2)


# ===== 第 4 段：缩放点积注意力（核心计算） =====
# 做什么：每个头独立执行 Attention(Q, K, V) = softmax(QK^T / sqrt(d)) * V
# 四步：1.点积算匹配度 -> 2.缩放防梯度消失 -> 3.softmax归一化 -> 4.加权求和
# 形状变化：见每步注释
# 涉及知识：[[缩放点积注意力]]（4步公式）、[[点积与向量相似度]]（matmul含义）

# --- 步骤1: 计算 Q 和 K 的点积，得到注意力原始权重 ---
# Q: (B, H, S, D) = (2, 2, 3, 2)
# K.transpose(-2, -1): 交换最后两维 -> (B, H, D, S) = (2, 2, 2, 3)
# matmul: (2, 2, 3, 2) x (2, 2, 2, 3) -> (2, 2, 3, 3)
# 含义：raw_weights[b][h][i][j] = 第b个样本、第h个头中，第i个词对第j个词的原始关注度
raw_weights = torch.matmul(Q, K.transpose(-2, -1))  # (2, 2, 3, 3)

# --- 步骤2: 缩放，除以 sqrt(head_dim)，防止点积值过大导致 softmax 梯度消失 ---
# 维度越大 -> 点积值越大 -> softmax 输出趋近 one-hot -> 梯度接近 0
# 除以 sqrt(d) 把方差拉回 1，让 softmax 在"舒适区"工作
scale_factor = K.size(-1) ** 0.5  # sqrt(head_dim) = sqrt(2) = 1.414

scaled_weights = raw_weights / scale_factor  # (2, 2, 3, 3) — 形状不变

# --- 步骤3: softmax 归一化，将每行变成概率分布（和为1） ---
# dim=-1 表示对最后一维（被关注的词）做 softmax
# 效果：每个词对所有其他词的关注度变成百分比
attn_weights = F.softmax(scaled_weights, dim=-1)  # (2, 2, 3, 3)

# --- 步骤4: 用注意力权重对 V 加权求和 ---
# attn_weights: (B, H, S, S) = (2, 2, 3, 3) — 注意力概率
# V:            (B, H, S, D) = (2, 2, 3, 2) — 每个词的实际内容
# matmul: (2, 2, 3, 3) x (2, 2, 3, 2) -> (2, 2, 3, 2)
# 含义：每个词的输出 = 按注意力权重从所有词的 Value 中提取信息
attn_outputs = torch.matmul(attn_weights, V)  # (2, 2, 3, 2)


# ===== 第 5 段：多头合并（combine_heads） =====
# 做什么：把多头结果拼回原来的形状
# 为什么：后续层（如 FFN）期望输入是 (B, S, F) 格式
# 形状变化：(B, H, S, D) -> transpose -> (B, S, H, D) -> contiguous -> view -> (B, S, F)
# 涉及知识：[[多头注意力机制]]（combine详解）、[[PyTorch张量变换]]（contiguous）

def combine_heads(tensor, num_heads):
    """
    将多头格式合并回原始形状。
    输入: (batch, num_heads, seq_len, head_dim)
    输出: (batch, seq_len, feature_dim)
    """
    batch_size, num_heads, seq_len, head_dim = tensor.size()
    feature_dim = num_heads * head_dim  # 总特征维度 = 头数 * 每头维度

    # transpose(1, 2): (B, H, S, D) -> (B, S, H, D) — 把头维度移回去
    # contiguous(): transpose后内存不连续，view要求连续，所以先拷贝成连续的
    # view: (B, S, H, D) -> (B, S, F) — 把 (H, D) 合并回 F
    output = tensor.transpose(1, 2).contiguous().view(batch_size, seq_len, feature_dim)
    return output  # (batch, seq_len, feature_dim)


attn_outputs = combine_heads(attn_outputs, num_heads)  # (2, 3, 4) — 多头信息已合并


# ===== 第 6 段：输出线性投影 =====
# 做什么：对合并后的结果做最后一次线性变换（论文中的 W_O）
# 为什么：让模型学习如何组合不同头的信息，不是简单拼接而是加权混合
# 形状变化：(2, 3, 4) -> Linear -> (2, 3, 4)
linear_out = torch.nn.Linear(4, 4)  # W_O: 输出投影矩阵
attn_outputs = linear_out(attn_outputs)  # (2, 3, 4) — 最终输出

print(" 加权信息 :", attn_outputs)