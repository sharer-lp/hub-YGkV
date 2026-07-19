# Week02_2026Q2 — AI 学习综合练习

## 目录结构

```
Week02_2026Q2/
├── tutorial-sklearn-torch/      # PyTorch + scikit-learn 机器学习基础
│   ├── 02_布尔索引.py           # 倒排索引与布尔搜索引擎
│   ├── 04_torch基础使用.py      # PyTorch Tensor 基础操作
│   ├── 05_torch梯度计算.py      # 自动求导 / 梯度计算
│   ├── 06_sklearn线性回归.py    # scikit-learn 线性回归
│   ├── 06_torch线性回归.py      # PyTorch 线性回归（手动参数 + SGD）
│   ├── 06_torch线性回归2.py     # 完全手动梯度下降（无优化器）
│   ├── 07_torch线性回归.py      # PyTorch 线性回归（nn.Linear 模块）
│   ├── 07_全链接层.py           # 手动实现全连接层并验证 nn.Linear
│   └── 08_全链接网络.py         # 多层全连接网络（MLP）
│
├── tutorial-llm-client/         # LLM API 调用与高级特性
│   ├── 01_基础调用.py           # 基础 Chat Completion 调用
│   ├── 02_流式输出.py           # 流式输出 + 思维过程展示
│   ├── 03_模型参数.py           # 五大模型参数对比演示
│   ├── 04_Tools.py              # 工具调用（Function Calling）6种模式
│   └── 05_JsonMode.py           # 结构化 JSON 输出模式
│
├── tutorial-prompt/             # 提示工程（Prompt Engineering）
│   ├── 02_零样本提示.py         # Zero-shot Prompting
│   ├── 03_少样本提示.py         # Few-shot Prompting
│   ├── 04_思维链提示.py         # Chain-of-Thought (CoT)
│   ├── 05_自我一致性.py         # Self-Consistency（多次采样 + 多数投票）
│   └── 06_生成知识提示.py       # Generate Knowledge Prompting
│
├── src/                         # Streamlit 综合应用
│   ├── model.py                 # 数据模型层（Pydantic v2）
│   ├── client.py                # LLM 服务封装
│   ├── embedding.py             # 语义编码服务（sentence-transformers）
│   ├── invert_index.py          # 倒排索引搜索引擎核心
│   ├── streamlit_app.py         # 应用入口 + 导航仪表盘
│   └── pages/
│       ├── 01_倒排索引.py       # 搜索引擎交互 Demo
│       ├── 02_编码模型.py       # 语义编码与聚类可视化
│       ├── 03_机器学习.py       # 线性回归双框架对比
│       └── 04_FAQ检索.py        # FAQ 综合检索系统
│
├── asserts/                     # 数据与配置文件
│   ├── faq.json                 # 电商 FAQ 知识库（34条）
│   ├── llm.deepseek.env         # DeepSeek API 配置
│   ├── llm.qwen.env             # 阿里云 DashScope 兼容配置
│   ├── 爬虫-新闻标题.txt        # 新闻标题语料（~720条）
│   └── bge-small-zh-v1.5/       # 本地 BGE 语义编码模型
│
└── .claude/                     # Claude 配置
```

---

## 一、tutorial-sklearn-torch — 机器学习与深度学习基础

### 1. 布尔索引与搜索引擎 (`02_布尔索引.py`)

| 知识点 | 说明 |
|--------|------|
| **倒排索引 (Inverted Index)** | `dict[str, set[int]]` 词→文档ID集合，O(1) 查询 |
| **jieba 分词** | 中文文本切词 |
| **布尔查询解析** | `and/or/not` 转换为 Python set 运算 `&/|/-` |
| **搜索模式对比** | 全表扫描 O(n) vs 倒排索引 O(1) |
| **HTML 高亮** | 匹配关键词 `<span style="color:red">` 包裹 |
| **eval() 动态执行** | 用户查询字符串转可执行 Python 表达式 |

### 2. PyTorch 张量基础 (`04_torch基础使用.py`)

- **4种 Tensor 创建方式**：直接创建、从 NumPy 转换、从已有张量派生、指定形状
- **Tensor 属性**：`.shape`, `.dtype`, `.device`
- **GPU 加速**：`tensor.to("cuda")` + `torch.cuda.is_available()` 防护
- **矩阵运算**：逐元素乘、`@` 矩阵乘法、`.T` 转置、`torch.cat` 拼接
- **Tensor ↔ NumPy**：共享底层内存，修改一个影响另一个

### 3. 自动求导 (`05_torch梯度计算.py`)

- `requires_grad=True` 开启梯度追踪
- 计算图自动记录前向传播操作
- `loss.backward()` 反向传播计算梯度
- `x.grad` 存储梯度值

### 4. 线性回归 — 4种实现方式对比

| 文件 | 方法 | 关键模式 |
|------|------|---------|
| `06_sklearn线性回归.py` | scikit-learn 闭式解 | `LinearRegression().fit(X, y)` |
| `06_torch线性回归.py` | 手动参数 + SGD 优化器 | 前向→损失→`zero_grad`→`backward`→`step` |
| `06_torch线性回归2.py` | 完全手动梯度下降 | `a -= lr * a.grad` 展示优化器内部原理 |
| `07_torch线性回归.py` | `nn.Linear` 模块 | 标准 PyTorch 做法 + `model.eval()` |

**训练循环标准模式**：
```python
optimizer.zero_grad()  # 清空梯度（梯度默认累加）
loss = loss_fn(y_pred, y)
loss.backward()        # 反向传播
optimizer.step()       # 更新参数
```

### 5. 全连接层 (`07_全链接层.py`)

- **手动实现**：`torch.matmul(input, weight.T) + bias`
- **验证**：`torch.allclose()` 对比手动结果与 `nn.Linear`
- **维度**：`(batch, in_features) × (out_features, in_features).T + (out_features,)`

### 6. 多层全连接网络 (`08_全链接网络.py`)

- **两种构建方式**：`nn.Sequential`（声明式）vs 自定义 `nn.Module` 子类（OOP）
- **激活函数**：`nn.Sigmoid`（历史）、`nn.ReLU`（现代，引入非线性）
- **结构模式**：输入 → Linear → 激活 → Linear → 激活 → ... → Linear（输出层无激活）
- **前向验证**：`model(dummy_input)` 检查输出形状

---

## 二、tutorial-llm-client — LLM API 调用与高级特性

### 1. 基础调用 (`01_基础调用.py`)

- OpenAI SDK 初始化：`OpenAI(api_key=..., base_url=...)` 适配 DeepSeek API
- `chat.completions.create()` 标准同步调用
- 消息结构：`system` + `user` 角色
- DeepSeek 特有：`reasoning_effort`、`extra_body={"thinking": {...}}`

### 2. 流式输出 (`02_流式输出.py`)

- `stream=True` 逐 chunk 增量输出
- 空 choices 防护 + `hasattr` 检查可选属性
- `delta.reasoning_content` 实时显示模型思维过程
- `chunk.usage` 获取 Token 用量统计

### 3. 模型参数对比 (`03_模型参数.py`)

| 参数 | 作用 | 演示值 |
|------|------|--------|
| `temperature` | 随机性控制 | 0.0 / 0.7 / 1.5 |
| `max_tokens` | 输出长度限制 | 20 / 100 / 500 |
| `reasoning_effort` | 思考深度 | low / medium / high |
| `logprobs` | Token 级置信度 | `top_logprobs=3` |
| `stop` | 提前终止序列 | `["。"]` |

### 4. 工具调用 / Function Calling (`04_Tools.py` ~425行)

**工具定义**：JSON Schema 描述函数 → `FUNCTION_MAP` 分发映射

**6种调用模式**：
1. **单次工具调用** — 模型自主选择工具
2. **多轮循环调用** — `for turn in range(max_turns)` 处理多步依赖
3. **并行工具调用** — 一次返回多个 `tool_calls`
4. **强制指定工具** — `tool_choice={"type": "function", "function": {"name": "calculate"}}`
5. **禁用工具** — `tool_choice="none"`
6. **流式工具调用** — 手动累积 delta 中的工具调用信息

**关键模式**：
```python
# 消息历史管理
messages.append({"role": "assistant", "content": None, "tool_calls": ...})
for tc in tool_calls:
    result = run_tool_call(tc)
    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
```

### 5. JSON 模式 (`05_JsonMode.py`)

- `response_format={"type": "json_object"}` 强制 JSON 输出
- 鲁棒解析三层回退：
  1. `json.loads()` 直接解析
  2. 去除 markdown 代码块标记后重试
  3. 打印原始内容用于调试
- 支持嵌套 JSON、数组输出、流式 JSON

---

## 三、tutorial-prompt — 提示工程

### 提示技术进阶路径

| 技术 | 文件 | 核心思想 |
|------|------|---------|
| **零样本提示** | `02_零样本提示.py` | 仅靠清晰指令完成任务，无示例 |
| **少样本提示** | `03_少样本提示.py` | 提供 2-5 个示例示范任务模式 |
| **思维链 CoT** | `04_思维链提示.py` | 引导逐步推理（Zero-shot + Few-shot 两种） |
| **自我一致性** | `05_自我一致性.py` | 多次采样 + 多数投票，减少随机误差 |
| **生成知识提示** | `06_生成知识提示.py` | 两步走：先生成背景知识 → 再基于知识回答 |

### 关键技术细节

- **Zero-shot CoT**：提示末尾加"请逐步推理"
- **Few-shot CoT**：示例中包含完整推理链条
- **自我一致性流程**：`sample_answers → extract_answer → Counter.most_common()`
- **生成知识提示**：通过 `assistant` 角色注入中间生成的知识
- **A/B 对比函数**：直接回答 vs 生成知识后回答同屏对比

---

## 四、src — Streamlit 综合应用

### 架构分层

```
src/model.py         ← 数据模型层 (Pydantic v2)
src/client.py        ← LLM 服务封装
src/embedding.py     ← 语义编码服务
src/invert_index.py  ← 倒排索引核心算法
src/streamlit_app.py ← 应用入口 + 仪表盘导航
src/pages/           ← 四个功能页面
```

### 1. 数据模型层 (`model.py`)

- **Pydantic v2**：`BaseModel` + `Field(description=..., ge=..., le=...)`
- **Literal 类型**：`match_type: Literal["exact", "fuzzy", "semantic"]`
- **类方法工厂**：`FaqDataset.from_json()` 兼容 `list`/`dict` 输入
- **序列协议**：实现 `__len__` / `__getitem__` 使类可迭代
- **计算属性**：`types` 属性返回去重排序的 FAQ 类型列表

### 2. LLM 服务 (`client.py`)

- `LLMClient` 类封装 OpenAI SDK，读取 `.env` 环境变量
- 构造函数参数默认值回退到 `os.getenv()`
- `chat_stream()` 使用 `yield` 实现流式生成器
- `_make_messages()` 统一处理 `str` 和 `list[dict]` 两种输入

### 3. 编码服务 (`embedding.py`)

- **sentence-transformers** 加载 BGE 中文语义编码模型
- **余弦相似度**：L2 归一化后点积
- **批量编码**：统一处理单条/多条文本 → `np.ndarray`
- **高效矩阵相似度**：`vecs_a @ vecs_b.T` 计算所有向量对相似度

### 4. 倒排索引 (`invert_index.py`)

- 核心结构：`dict[str, set[int]]`
- 手写查询编译器：jieba 分词 → set 表达式 → `eval()`
- 短语匹配：缓存连续分词为短语再查询
- NOT 运算符：全集减匹配集合

### 5. 应用入口 (`streamlit_app.py`)

- **多页面路由**：Streamlit `pages/` 目录约定
- **自定义 CSS** 注入：卡片悬停效果、统计数字样式
- **仪表盘布局**：5列统计卡片 + 3列导航卡片
- `sys.path` 处理确保模块导入

### 6. 功能页面

#### 01_倒排索引.py — 搜索引擎 Demo
- `st.session_state` 缓存索引实例避免重构建
- `st.expander` 渐进展开统计信息
- 布尔查询 UI + 操作符帮助表
- 词项频次分布柱状图（matplotlib）
- 查询结果 HTML 高亮渲染

#### 02_编码模型.py — 语义编码与聚类可视化
- `@st.cache_resource` 缓存大模型加载
- 三 Tab：语义编码 / 相似度计算 / 聚类可视化
- **KMeans 聚类** + **PCA 两步降维**（512→50→2D）
- **轮廓系数** 评估聚类质量
- 相似度矩阵热力图（YlOrRd 色图 + 数值标注）
- Pandas DataFrame 渐变样式

#### 03_机器学习.py — 线性回归双框架对比
- scikit-learn vs PyTorch 解决同一问题
- `st.progress()` 实时训练进度条
- 训练历史记录 + 参数收敛跟踪表格
- 双面板可视化：损失曲线 + 拟合直线
- `st.metric(delta=...)` 显示与真实值的偏差
- 中文字体跨平台配置

#### 04_FAQ检索.py — 综合检索系统
- **BM25 全文检索**（`bm25s` 库）
- 语义检索 + BM25 双 Tab 对比
- **LLM 零样本分类**：预测 FAQ 问题类型
- **JSON 解析三层回退**：
  1. `json.loads()` 直接解析
  2. 正则提取 `r'"type"\s*:\s*"([^"]+)"'`
  3. 关键词匹配已知类型
- 惰性加载 embedding 模型（仅首次查询时初始化）
- 结果按相似度降序排列 + 彩色进度条

---

## 五、asserts — 数据与配置文件

| 文件 | 格式 | 内容与用途 |
|------|------|-----------|
| `faq.json` | JSON (34 条) | 电商客服知识库，含售后/订单/客服/优惠/账户/配送 6 类 |
| `llm.deepseek.env` | 环境变量 | DeepSeek API 连接配置 |
| `llm.qwen.env` | 环境变量 | 阿里云 DashScope 兼容模式配置 |
| `爬虫-新闻标题.txt` | 纯文本 (~720 条) | 新闻标题 NLP 语料（社会/娱乐/体育/财经等） |
| `bge-small-zh-v1.5/` | 模型文件 | 本地 BGE 中文语义编码模型 |

---

## 六、技术栈全景

| 领域 | 库/工具 | 核心知识点 |
|------|---------|-----------|
| **深度学习框架** | `torch`, `torch.nn`, `torch.optim` | Tensor 运算、自动求导、Linear 层、MLP、SGD |
| **传统机器学习** | `sklearn` | 线性回归闭式解、KMeans 聚类、PCA 降维、R²/MSE |
| **LLM API** | `openai` SDK | Chat Completion、流式输出、Tools、JSON Mode、参数调优 |
| **向量编码** | `sentence-transformers`, `numpy` | BGE 模型、余弦相似度、矩阵运算 |
| **全文检索** | `jieba`, `bm25s` | 中文分词、倒排索引、BM25 排序、布尔查询 |
| **数据建模** | `pydantic` v2 | Schema 验证、Literal 类型、序列协议 |
| **提示工程** | (纯 LLM 交互) | Zero/Few-shot、CoT、Self-Consistency、知识生成 |
| **交互界面** | `streamlit`, `matplotlib`, `pandas` | 多页面、热力图、聚类散点图、实时进度条 |

---

## 七、CLI 命令速查

```bash
# 启动 Streamlit 应用
cd /Users/lyz/Work/八斗学院/Week02_2026Q2
streamlit run src/streamlit_app.py

# 运行单个教程脚本（需先配置 API Key）
cd tutorial-prompt && python 02_零样本提示.py
cd tutorial-llm-client && python 01_基础调用.py
cd tutorial-sklearn-torch && python 04_torch基础使用.py
```

> **注意**：运行 LLM 相关脚本前，需在 `asserts/` 下配置 `.env` 文件中的 API Key。
