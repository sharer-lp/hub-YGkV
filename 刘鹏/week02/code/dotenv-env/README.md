# dotenv-env：多环境 + 多模型 生产级配置与客户端封装

> 基于 `python-dotenv` + `openai` SDK 的 LLM 调用示例，支持多环境切换、多模型管理、思考模式、流式输出。

## 环境要求

- Python ≥ 3.10
- 依赖安装：

```bash
pip install openai python-dotenv
```

## 快速开始

```bash
# 1. 复制模板并填入你的 API Key
cp .env.example .env
cp .env.example .env.models

# 2. 运行示例
python main.py
```

## 文件结构

```
dotenv-env/
├── .env              ← 通用默认配置（所有环境共享）
├── .env.dev          ← 开发环境覆盖（DEBUG=true, 宽松超时）
├── .env.prod         ← 生产环境覆盖（DEBUG=false, 严格参数）
├── .env.models       ← 多模型密钥与连接信息（独立管理）
├── .env.example      ← 配置模板（提交到 Git）
├── config.py         ← 配置加载引擎，暴露全局单例 cfg
├── deepseek_client.py← LLM 客户端封装（兼容所有 OpenAI 协议模型）
└── main.py           ← 运行入口
```

## 核心设计：多文件加载优先级

```
系统环境变量（最高优先级）
    ↓ 覆盖
.env.{APP_ENV}（如 .env.prod）
    ↓ 覆盖
.env.models（模型密钥）
    ↓ 覆盖
.env（通用默认值，最低优先级）
```

**关键原则**：`override=False` 保证系统环境变量始终最高优先级（生产安全实践）。

## 环境切换

| 方式 | 命令 | 适用场景 |
|------|------|----------|
| PowerShell 临时 | `$env:APP_ENV="prod"; python main.py` | 本地测试生产配置 |
| Linux/macOS 临时 | `APP_ENV=prod python main.py` | 同上 |
| 系统环境变量 | 在系统设置中永久配置 | 服务器部署 |
| .env 文件 | 修改 `APP_ENV=prod` | 不推荐（失去隔离意义） |

## 多模型管理

### 配置方式（.env.models）

```bash
# 命名规则：{别名大写}_API_KEY / _BASE_URL / _MODEL
DEEPSEEK_API_KEY=sk-xxx
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash

QWEN_API_KEY=sk-xxx
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen-plus

# 切换模型只改这一行
ACTIVE_MODEL=deepseek
```

### 代码中切换模型

```python
from deepseek_client import DeepSeekClient

# 使用默认激活模型
client = DeepSeekClient()

# 显式指定模型
client = DeepSeekClient(model_name="qwen")
```

### 命令行临时切换

```powershell
$env:ACTIVE_MODEL="qwen"; python main.py
```

## 配置变量一览

### 通用配置（.env）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `APP_ENV` | `dev` | 环境标识：`dev` / `prod` |
| `ENABLE_THINKING` | `false` | 是否开启思考模式 |
| `REASONING_EFFORT` | `high` | 思考强度：`low`/`medium`/`high`/`max` |
| `MAX_TOKENS` | `4096` | 最大输出 token 数 |
| `TIMEOUT` | `120` | 请求超时（秒） |
| `MAX_RETRIES` | `3` | 自动重试次数 |
| `DEBUG` | `false` | 调试模式 |
| `LOG_LEVEL` | `INFO` | 日志级别 |

### 模型配置（.env.models）

| 变量 | 说明 |
|------|------|
| `{ALIAS}_API_KEY` | 模型 API 密钥（必填） |
| `{ALIAS}_BASE_URL` | API 地址 |
| `{ALIAS}_MODEL` | 模型名称 |
| `ACTIVE_MODEL` | 当前激活的模型别名 |

## 思考模式说明

DeepSeek V4 通过 `thinking` 参数控制：

```python
extra_body = {"thinking": {"type": "enabled"}}   # 开启
extra_body = {"thinking": {"type": "disabled"}}  # 关闭
```

- 思考模式下 **不支持** `temperature` / `top_p`
- 多轮对话拼接时 **必须剥离** `reasoning_content`

## 注意事项

1. **安全**：`.env` / `.env.dev` / `.env.prod` / `.env.models` 均已加入 `.gitignore`，切勿提交
2. **协作**：只提交 `.env.example`，团队成员复制后填入自己的密钥
3. **代理**：若设置了 `HTTP_PROXY`/`HTTPS_PROXY`，SDK 会自动读取导致连接失败，临时清除：`$env:HTTPS_PROXY=""`
4. **扩展模型**：在 `config.py` 的 `KNOWN_MODELS` 列表中添加别名，然后在 `.env.models` 中配置对应变量即可
