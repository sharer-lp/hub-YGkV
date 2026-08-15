# 10_BERT文本分类.py 运行与调参指南

本文档针对 `E:\code\hub-YGkV\刘鹏\Week04\10_BERT文本分类.py`，说明如何把脚本跑起来，并介绍常见超参数怎么改、怎么对比实验结果。

## 1. 脚本在做什么

- 读取 `../Week01/dataset.csv`：Tab 分隔、无表头，第 1 列是文本，第 2 列是类别标签。
- 只取前 500 条数据，按 8:2 分层划分训练集和测试集。
- 从 `../../models/google-bert/bert-base-chinese`（仓库根目录的 `models`）加载 BERT 中文预训练模型。
- 用 Hugging Face `Trainer` 训练 4 个 epoch、batch size 16，每个 epoch 结束评估一次。
- 最终通过 `trainer.evaluate()` 输出 `eval_accuracy`。

## 2. 当前环境检查结果

| 项目 | 结果 |
| --- | --- |
| 推荐解释器 | `D:\App_Library\Miniconda3\envs\py312\python.exe`，即 conda 环境 `py312` |
| 依赖 | torch 2.6.0+cpu、transformers 4.55.0、datasets 5.0.1、pandas 2.2.2、scikit-learn 1.5.1 |
| GPU | 无，`torch.cuda.is_available()` 为 False，只能 CPU 训练 |
| 数据文件 | `E:\code\hub-YGkV\刘鹏\Week01\dataset.csv` 不存在 |
| 模型目录 | `E:\code\hub-YGkV\models\google-bert\bert-base-chinese` 不存在 |

注意：默认的 `D:\App_Library\Miniconda3\python.exe`（base 环境）缺少 `pandas`、`datasets`、`scikit-learn`，所以运行请使用 `py312` 环境。

## 3. 准备数据（二选一）

### 方案 A：使用仓库里已有的意图分类数据（最快）

仓库中已经有现成数据：`E:\code\hub-YGkV\刘鹏\week03\code\01-intent-classify\assets\dataset\dataset.csv`，共 12 类意图。

把脚本里第 13 行和第 30/31 行改成：

```python
dataset_df = pd.read_csv("../week03/code/01-intent-classify/assets/dataset/dataset.csv", sep="\t", header=None)
```

```python
model = BertForSequenceClassification.from_pretrained('../../models/google-bert/bert-base-chinese', num_labels=12)
```

说明：`10_BERT文本分类.py` 是从 week03 的 `train_bert.py` 复制改来的，原脚本就是使用这份 12 类数据和 `num_labels=12`。Week04 这份被改成了 17 类新闻任务的写法，但仓库里还没有对应的 17 类数据。

路径说明：脚本在 `刘鹏\Week04` 下运行，所以 `../week03/...` 指向 `刘鹏\week03\...`；模型按第 4 节下载在仓库根目录 `E:\code\hub-YGkV\models\google-bert\bert-base-chinese`，因此脚本里要用 `../../models/google-bert/bert-base-chinese`（向上两级），不要用 `../models/...`，否则会去找不存在的 `刘鹏\models\...`。

### 方案 B：准备脚本预期的 17 类新闻数据

1. 把数据放到 `E:\code\hub-YGkV\刘鹏\Week01\dataset.csv`。
2. 格式必须是两列、Tab 分隔、无表头：

```text
文本内容1	体育
文本内容2	科技
```

3. 数据总量至少 500 行，并且前 500 行里要覆盖全部类别，否则 `stratify=labels` 会报错。
4. 把第 31 行的 `num_labels=17` 改成根据数据自动计算，避免类别数不匹配：

```python
model = BertForSequenceClassification.from_pretrained('模型路径', num_labels=len(lbl.classes_))
```

也可以在训练前打印检查：

```python
print(lbl.classes_)
print("类别数量:", len(lbl.classes_))
```

## 4. 下载预训练模型（必做）

推荐使用 ModelScope（国内访问更快）。在仓库根目录执行：

```powershell
cd E:\code\hub-YGkV
D:\App_Library\Miniconda3\envs\py312\python.exe -m pip install -U modelscope
D:\App_Library\Miniconda3\envs\py312\Scripts\modelscope.exe download --model google-bert/bert-base-chinese --local_dir models/google-bert/bert-base-chinese
```

注意：新版 ModelScope 没有 `python -m modelscope` 入口，直接执行 `modelscope.exe`（已激活 `py312` 时也可以直接写 `modelscope download ...`）。

如果命令行仍报错，可以用 Python API 方式下载：

```powershell
cd E:\code\hub-YGkV
D:\App_Library\Miniconda3\envs\py312\python.exe -c "from modelscope import snapshot_download; snapshot_download('google-bert/bert-base-chinese', local_dir='models/google-bert/bert-base-chinese')"
```

如果使用 Hugging Face，也可以：

```powershell
cd E:\code\hub-YGkV
D:\App_Library\Miniconda3\envs\py312\Scripts\huggingface-cli.exe download google-bert/bert-base-chinese --local-dir models/google-bert/bert-base-chinese
```

下载完成后确认目录里有这些文件：

```text
config.json
tokenizer_config.json
vocab.txt
pytorch_model.bin
```

`07_huggingface.py` 也使用同一个模型路径，下载一次即可共用。

## 5. 检查依赖

```powershell
conda activate py312
python -c "import pandas, sklearn, datasets, transformers, torch; print('依赖 OK')"
```

如果报错，再安装：

```powershell
pip install -U pandas scikit-learn datasets transformers
```

当前机器的 `py312` 环境已经满足要求，通常不需要执行安装。

## 6. 运行脚本

```powershell
cd E:\code\hub-YGkV\刘鹏\Week04
D:\App_Library\Miniconda3\envs\py312\python.exe 10_BERT文本分类.py
```

激活环境后也可以直接写：

```powershell
conda activate py312
cd E:\code\hub-YGkV\刘鹏\Week04
python 10_BERT文本分类.py
```

运行完成后，结果会在：

- `E:\code\hub-YGkV\刘鹏\Week04\results`：模型 checkpoint。
- `E:\code\hub-YGkV\刘鹏\Week04\logs`：训练日志。
- 终端输出：`trainer.evaluate()` 的字典，例如 `{'eval_loss': ..., 'eval_accuracy': 0.85, ...}`。

### CPU 快速冒烟测试

500 条数据、batch 16、4 个 epoch 大约 100 个训练步。BERT 在 CPU 上训练会很慢，首次运行建议先只跑两步验证流程：

在 `TrainingArguments(...)` 里临时加上：

```python
max_steps=2,
report_to=[],
```

能正常训练并输出 `eval_accuracy` 后再删掉 `max_steps`，正式跑完整训练。

## 7. 超参数在哪里改、怎么测

### 7.1 参数清单

| 参数 | 位置 | 当前值 | 建议范围 | 说明 |
| --- | --- | --- | --- | --- |
| `learning_rate` | `TrainingArguments` 里未写，可新增 | 默认 5e-5 | 1e-5 到 5e-5 | BERT 微调常用 2e-5 |
| `num_train_epochs` | `TrainingArguments` | 4 | 2 到 6 | 小数据集轮数太多容易过拟合 |
| `per_device_train_batch_size` | `TrainingArguments` | 16 | 8 到 32 | CPU 建议小 batch，显存不够就调小 |
| `per_device_eval_batch_size` | `TrainingArguments` | 16 | 8 到 32 | 同上 |
| `warmup_steps` | `TrainingArguments` | 500 | 0 或总步数 10% 左右 | 当前 500 已经大于总训练步数，预热过长 |
| `weight_decay` | `TrainingArguments` | 0.01 | 0 到 0.1 | 越大正则化越强 |
| `logging_steps` | `TrainingArguments` | 100 | 10 到 100 | 日志打印频率 |
| `eval_strategy` | `TrainingArguments` | `epoch` | `epoch` 或 `steps` | 小数据按 epoch 评估即可 |
| `max_length` | tokenizer 编码处 | 64 | 32 到 128 | 文本更长时可调大，但训练更慢 |
| `test_size` | `train_test_split` | 0.2 | 0.15 到 0.3 | 测试集比例 |
| 数据量 | `[:500]` | 500 | 100 到全部 | 先用小数据验证，再逐步加大 |

想显式设置学习率，在 `TrainingArguments` 中加一行：

```python
learning_rate=2e-5,
```

建议同时加上这几个参数，让实验更干净：

```python
metric_for_best_model="accuracy",
save_total_limit=2,
report_to=[],
```

### 7.2 测试方法

1. 每次只改一个超参数，其他参数保持不变。
2. 记录实验名、参数值和最终 `eval_accuracy`。
3. 对比时优先看测试集 accuracy，而不是训练 loss。

可以按下面的表格记录：

| 实验 | learning_rate | epochs | batch | max_length | warmup_steps | eval_accuracy | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 2e-5 | 3 | 16 | 64 | 10 | ? | 初始配置 |
| lr_1e-5 | 1e-5 | 3 | 16 | 64 | 10 | ? | 只改学习率 |
| batch_8 | 2e-5 | 3 | 8 | 64 | 10 | ? | 只改 batch |
| max_32 | 2e-5 | 3 | 16 | 32 | 10 | ? | 只改 max_length |

建议测试顺序：先固定 `learning_rate=2e-5`，依次测 epoch、batch size、`max_length`、`warmup_steps`，最后再单独测学习率。

### 7.3 可选：改成命令行传参

如果不想每次打开脚本改数字，可以在脚本开头加 `argparse`，把最常用的参数提出来：

```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", default="../Week01/dataset.csv")
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--max_length", type=int, default=64)
args = parser.parse_args()
```

然后把 `read_csv`、tokenizer 和 `TrainingArguments` 中的对应值替换为 `args.data_path`、`args.lr`、`args.epochs`、`args.batch_size`、`args.max_length`。之后就可以这样测试：

```powershell
python 10_BERT文本分类.py --lr 5e-5 --epochs 2 --batch_size 8 --max_length 32
```

## 8. 常见报错

| 报错 | 原因 | 解决 |
| --- | --- | --- |
| `FileNotFoundError: ../Week01/dataset.csv` | 数据文件不存在 | 按第 3 节准备数据或改路径 |
| `OSError ... bert-base-chinese` | 本地模型没下载 | 按第 4 节下载模型 |
| `Target size ... must be the same as input size` | `num_labels` 和真实类别数不一致 | 改成 `len(lbl.classes_)` |
| `ModuleNotFoundError: No module named 'datasets'/'pandas'` | 用了 base 环境 | 使用 `py312` 环境 |
| 弹出 wandb 相关提示或日志刷屏 | 默认开启了实验报告 | `TrainingArguments` 加 `report_to=[]` |
| 训练很慢 | CPU 训练 BERT | 先用 `max_steps=2` 冒烟，再缩小数据量或 `max_length` |

## 9. 和 week03 原脚本的关系

`10_BERT文本分类.py` 是 `E:\code\hub-YGkV\刘鹏\week03\code\01-intent-classify\app\training_code\train_bert.py` 的简化版。原脚本用绝对路径读取 week03 的 12 类意图数据，并把最优模型保存为 `assets/weights/bert.pt`。如果只想验证意图分类流程，可以直接运行原脚本；Week04 这份脚本更像是一个独立的 BERT 分类练习，需要自行准备好它引用的数据和模型。
