---
title: BERT 微调超参数与训练日志学习笔记
tags:
  - NLP
  - BERT
  - 微调
  - 超参数
  - 训练日志
  - AI应用开发
aliases:
  - BERT超参数笔记
  - 训练日志字段笔记
created: 2026-08-15
updated: 2026-08-15
source: 10_BERT文本分类.py
related:
  - "[[实验日志分析报告]]"
  - "[[调试文档_新手版]]"
  - "[[logs_to_table]]"
---

# BERT 微调超参数与训练日志学习笔记

> [!info] 这份笔记适合谁
> AI 应用开发方向，BERT 微调脚本已经能跑通，想理解“日志在说什么、参数怎么调、什么时候调什么”。

## 1. 当前配置是否覆盖了所有常用参数

> [!important] 结论
> 当前配置覆盖了 BERT 微调约 90% 的常用参数，但不是全部。

### 已覆盖的核心参数

```python
output_dir='./results'
num_train_epochs=3
per_device_train_batch_size=16
per_device_eval_batch_size=16
warmup_steps=10
learning_rate=2e-5
weight_decay=0.01
max_grad_norm=1.0
logging_dir='./logs'
logging_steps=10
eval_strategy="epoch"
save_strategy="epoch"
load_best_model_at_end=True
metric_for_best_model="accuracy"
save_total_limit=3
report_to=["tensorboard"]
```

### 还没有覆盖但常用的参数

| 参数 | 作用 | 什么时候用 |
| --- | --- | --- |
| `lr_scheduler_type` | 学习率调度方式 | 想换余弦衰减、常数学习率时 |
| `warmup_ratio` | 按比例预热 | 数据量经常变化时更稳定 |
| `label_smoothing_factor` | 标签平滑 | 过拟合、模型过于自信时 |
| `gradient_accumulation_steps` | 梯度累积 | 显存不足又不想减小 batch |
| `seed` | 随机种子 | 需要实验可复现时 |
| `max_steps` | 最大训练步数 | 冒烟测试、限制训练时长 |
| `fp16` / `bf16` | 混合精度 | 有 GPU 时加速 |
| `gradient_checkpointing` | 梯度检查点 | 显存不足时 |
| `dataloader_num_workers` | 数据加载进程数 | 数据加载成为瓶颈时 |
| `optim` | 优化器 | 默认 AdamW 就够 |
| `hidden_dropout_prob` | BERT 内部 dropout | 过拟合时 |
| 早停回调 | 提前停止训练 | 防止无效训练和过拟合 |

## 2. 训练日志每个字段的含义

### 2.1 训练过程字段

| 字段 | 含义 | 深层意义 | 异常时怎么调 |
| --- | --- | --- | --- |
| `loss` | 当前训练损失 | 越小说明模型在训练集上越接近正确答案 | 一直不降：检查 lr、数据、warmup |
| `grad_norm` | 梯度范数 | 衡量参数更新力度，过大说明训练震荡 | 持续大于 10：加 `max_grad_norm=1.0` 或降 lr |
| `learning_rate` | 当前实际学习率 | 不是固定值，会预热再衰减 | 结束前接近 0：减 epochs 或增加数据 |
| `epoch` | 当前训练轮次 | 一个 epoch = 把所有训练数据学一遍 | 看日志属于第几轮 |

### 2.2 评估字段

| 字段 | 含义 | 深层意义 | 异常时怎么调 |
| --- | --- | --- | --- |
| `eval_loss` | 测试集损失 | 越低说明泛化越好 | 回升：过拟合，减 epochs / 加正则 |
| `eval_accuracy` | 测试集准确率 | 模型真实能力的最直接指标 | 卡住：扩数据；波动大：小测试集正常 |
| `eval_runtime` | 评估耗时 | 只影响速度，不影响质量 | 明显变慢：可能系统负载高 |
| `eval_samples_per_second` | 每秒评估样本数 | 评估速度 | 不用作调参依据 |
| `eval_steps_per_second` | 每秒评估 batch 数 | 评估速度 | 不用作调参依据 |

### 2.3 最终汇总字段

| 字段 | 含义 | 深层意义 |
| --- | --- | --- |
| `train_runtime` | 总训练耗时 | 用来估算扩数据后的成本 |
| `train_samples_per_second` | 每秒训练样本数 | 衡量训练速度 |
| `train_steps_per_second` | 每秒训练步数 | 衡量训练速度 |
| `train_loss` | 训练阶段平均损失 | 和 `eval_loss` 一起判断是否过拟合 |

## 3. 日志特征对应什么调整

| 日志特征 | 说明 | 优先调整 |
| --- | --- | --- |
| accuracy 卡在 0.1 左右 | 模型基本没学会 | 检查类别数、数据是否打乱、lr、warmup |
| accuracy 提升后停滞 | 数据量可能不够 | 扩数据，而不是继续加 epoch |
| train_loss 一直降，eval_loss 回升 | 过拟合 | 减 epochs、早停、weight_decay、dropout |
| grad_norm 持续大于 10 | 训练震荡 | `max_grad_norm=1.0`，必要时降 lr |
| learning_rate 接近 0 还有 epoch | 后面在空转 | 减 epochs，或增加数据让步数变多 |
| eval_runtime 忽快忽慢 | 系统负载波动 | 忽略，不影响模型 |
| eval_accuracy 波动 1-2 个点 | 测试集太小 | 用 F1，多次实验取平均 |

## 4. 超参数学习笔记

### 4.1 learning_rate

- 含义：每次参数更新走多大一步。
- 当前建议：`2e-5`
- 范围：`1e-5` 到 `5e-5`
- 场景：
  - 模型学不动：尝试 `5e-5`。
  - 训练震荡：降到 `1e-5`。
  - 过拟合：降低学习率并配合早停。

### 4.2 num_train_epochs

- 含义：全部训练数据被学习几遍。
- 当前建议：`3`
- 范围：`2` 到 `6`
- 场景：
  - 小数据集：2 到 3 足够。
  - 数据量增大：可以增加到 4 到 5。
  - accuracy 开始下降：说明已经过头。

### 4.3 per_device_train_batch_size

- 含义：每个 batch 一次喂给模型多少条样本。
- 当前建议：`16`
- 范围：`8` 到 `32`
- 场景：
  - 显存不足：调小。
  - CPU 太慢：调小不一定更快，先降 max_length。
  - 训练不稳定：尝试 8。

### 4.4 per_device_eval_batch_size

- 含义：评估时每个 batch 多少条。
- 当前建议：`16`
- 范围：`8` 到 `32`
- 场景：通常和训练 batch 保持一致即可。

### 4.5 warmup_steps / warmup_ratio

- 含义：前多少步把学习率从接近 0 升到设定值。
- 当前建议：`warmup_steps=10`
- 范围：总步数的 5% 到 10%。
- 场景：
  - 总步数小：用 `warmup_steps=10`。
  - 数据量经常变：用 `warmup_ratio=0.1` 更省心。
  - 日志里 lr 一直很小：warmup 太大。

### 4.6 lr_scheduler_type

- 含义：学习率随训练怎么变化。
- 常用值：`linear`、`cosine`、`constant`。
- 场景：
  - 默认线性衰减最常用。
  - 想让学习率下降更平滑：`cosine`。
  - 只想固定学习率：`constant`。

### 4.7 weight_decay

- 含义：正则化强度，越大越不容易记住训练集细节。
- 当前建议：`0.01`
- 范围：`0` 到 `0.1`
- 场景：
  - 过拟合：升到 `0.05` 或 `0.1`。
  - 正常训练：保持 `0.01`。

### 4.8 max_grad_norm

- 含义：梯度范数上限，防止参数更新过大。
- 当前建议：`1.0`
- 范围：`0.5` 到 `5.0`
- 场景：
  - 日志 grad_norm 大于 10：设 `1.0`。
  - 仍然震荡：降到 `0.5` 或同时降 lr。

### 4.9 label_smoothing_factor

- 含义：让模型不要过于自信地贴近训练标签。
- 常用值：`0.1`
- 范围：`0` 到 `0.2`
- 场景：过拟合、模型输出概率过于极端时。

### 4.10 max_length

- 含义：文本被截断或填充到的最大 token 数。
- 当前建议：`32`
- 范围：`16` 到 `128`
- 场景：
  - 文本平均长度远小于 max_length：调小，训练更快。
  - 文本长且信息在后半段：调大。
  - 用日志分析文本 P95，选比 P95 略大的值。

### 4.11 seed

- 含义：随机种子，控制数据打乱和参数初始化。
- 建议：固定 `seed=42`。
- 场景：需要对比实验、复现结果时必设。

### 4.12 gradient_accumulation_steps

- 含义：累积几个 batch 的梯度再更新一次参数。
- 场景：显存不足但不想减 batch 时。
- 注意：等效 batch = batch_size x accumulation。

### 4.13 eval_strategy / save_strategy

- 含义：什么时候评估、什么时候保存。
- 常用值：`epoch` 或 `steps`。
- 场景：小数据用 `epoch`；大数据或想更细粒度看曲线用 `steps`。

### 4.14 load_best_model_at_end

- 含义：训练结束后加载历史上最好的 checkpoint。
- 前提：必须配 `metric_for_best_model`。
- 场景：只要做调参对比就打开。

### 4.15 save_total_limit

- 含义：最多保留几个 checkpoint。
- 建议：`3` 到 `5`。
- 原因：设为 1 时，历史最优 checkpoint 可能被删掉。

### 4.16 dropout

- 含义：训练时随机丢弃部分神经元。
- 位置：`BertConfig.hidden_dropout_prob` 和 `attention_probs_dropout_prob`。
- 默认：约 0.1。
- 场景：过拟合时可提到 `0.2` 到 `0.3`。

## 5. 不同场景速查表

| 场景 | 优先调整 | 次要调整 |
| --- | --- | --- |
| 模型没学会 | 检查数据、lr=2e-5、warmup=10 | 增加 max_length |
| 过拟合 | 减 epochs、早停 | weight_decay、dropout、label smoothing |
| 训练震荡 | max_grad_norm=1.0 | 降 lr |
| 训练太慢 | 减 max_length、减数据量 | 开多进程、用 GPU |
| 显存不足 | 减 batch、减 max_length | gradient_accumulation、gradient_checkpointing |
| 小数据集 | 少 epochs、小 lr、强正则 | 扩数据 |
| 大数据集 | 可增加 epochs | 用 warmup_ratio、cosine 调度 |
| 类别不平衡 | stratify、看 F1 | 增加少数类样本 |
| 文本很长 | 增大 max_length | 减 batch 控制显存 |
| 文本很短 | 减小 max_length | 训练更快 |

## 6. 数据变化对参数的影响

| 数据变化 | 影响 | 怎么调 |
| --- | --- | --- |
| 数据量增大 | 更难过拟合，训练更慢 | epochs 可增加，warmup_ratio 更稳定 |
| 数据量减小 | 更容易过拟合 | 减 epochs、加正则、降 lr |
| 类别更不平衡 | accuracy 会骗人 | stratify，重点看 F1 |
| 文本变长 | 训练变慢、占内存 | 增大 max_length，或减 batch |
| 文本变短 | 大量 padding 浪费算力 | 减小 max_length |
| 数据噪声变大 | 模型学不稳定 | 降 lr、早停、加 weight_decay |
| 数据分布变化 | 旧模型失效 | 重新训练或做领域适配 |

## 7. 推荐调参流程

1. 先保证数据正确：打乱、stratify、类别齐全。
2. 跑一次 baseline，记录到 `experiments.csv`。
3. 先调 learning_rate，再调 epochs。
4. 出现过拟合，再调 weight_decay、dropout、label smoothing。
5. 每次只改一个参数。
6. 用 F1 而不是只看 accuracy。
7. 固定 seed，多跑几次取平均。

> [!tip] 调参铁律
> 一次只改一个参数，其他全部保持不变；先小数据验证，再全量训练。

## 8. 记录和可视化

- `experiments.csv`：每次实验自动追加一行，用 Excel 对比。
- `logs_to_table.py`：把 `trainer_state.json` 转成 Markdown / CSV / Excel。
- TensorBoard：看 loss、lr、grad_norm、eval 曲线。
- MLflow：了解即可，本地小项目暂不启用。

## 9. Obsidian 使用提示

- 本笔记已带 YAML frontmatter，Obsidian 会识别标题、标签、创建时间。
- 侧边栏打开关系图谱，可以看到 `[[实验日志分析报告]]` 等关联笔记。
- 可以把本笔记加入 MOC（目录笔记）：

```markdown
- [[BERT微调超参数与训练日志学习笔记]]
- [[实验日志分析报告]]
- [[调试文档_新手版]]
```
