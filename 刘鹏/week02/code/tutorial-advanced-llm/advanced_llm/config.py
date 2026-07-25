"""
生产级多环境 + 多模型 + 多格式 配置管理

==================== 架构设计 ====================

【本文件职责】
    集中管理所有配置项，对外暴露一个全局单例 `cfg`。
    其他模块只需 `from advanced_llm.config import cfg` 即可获取任何配置。

【核心设计模式】
    1. 单例模式（Singleton）：
       - 文件末尾 `cfg = Config()` 创建唯一实例
       - 整个进程生命周期内只初始化一次，所有模块共享同一份配置
       - 避免重复加载 .env 文件、重复解析环境变量

    2. 分层配置（Layered Configuration）：
       - 类似 CSS 的层叠规则：后定义的覆盖先定义的
       - 加载优先级（后加载的覆盖先加载的）：
         系统环境变量 > .env.{APP_ENV} > .env.models > .env

【@dataclass 装饰器详解】
    @dataclass 是 Python 3.7+ 引入的语法糖，用最少代码定义"数据容器类"。
    它会自动生成以下魔术方法：
        - __init__()   构造器（根据字段自动生成）
        - __repr__()   打印时的字符串表示
        - __eq__()     对象相等比较（逐字段对比）

    常用参数：
        @dataclass(
            repr=True,       # 生成 __repr__（默认 True）
            eq=True,         # 生成 __eq__（默认 True）
            frozen=True,     # 让对象不可变（赋值会抛 FrozenInstanceError）
            order=True,      # 生成 __lt__、__le__、__gt__、__ge__ 排序方法
            unsafe_hash=True # 强制生成 __hash__（让对象可作为字典 key）
        )

    本项目中 ModelConfig 使用 frozen=True：
        - 配置对象创建后不可修改 → 防止运行时意外篡改
        - 线程安全 → 多线程读取无需加锁

【环境切换方式】
    方式1：系统环境变量  APP_ENV=prod python main.py（Linux/Mac）
    方式2：PowerShell    $env:APP_ENV="prod"; python main.py（Windows）
    方式3：修改 .env 文件中的 APP_ENV=prod

【多模型切换】
    .env 或 .env.models 中修改 ACTIVE_MODEL=deepseek / qwen / kimi / claude
    或系统环境变量 ACTIVE_MODEL=qwen python main.py

【多格式支持】
    每个模型通过 {MODEL}_PROVIDER 指定 API 格式：
        - openai    → OpenAI Chat Completions 格式（DeepSeek / Qwen / Kimi）
        - anthropic → Anthropic Messages 格式（Claude）
    不同格式的协议差异由 clients/ 下的具体客户端屏蔽，上层代码无感知。

==================================================
"""

# ==================== 导入区 ====================
# 【知识点：标准库 vs 第三方库】
#   os / pathlib / dataclasses → Python 标准库（无需 pip install）
#   dotenv → 第三方库 python-dotenv（需要 pip install python-dotenv）

import os                          # 操作系统接口：读环境变量、文件路径等
from pathlib import Path           # 面向对象的路径操作（比 os.path 更现代）
from dataclasses import dataclass, field  # 数据类装饰器
from dotenv import load_dotenv     # 将 .env 文件内容加载到 os.environ


# ==================== 异常 ====================
# 【知识点：自定义异常】
#   继承 Exception 即可创建自定义异常类。
#   好处：调用方可以精确捕获 "配置错误" 而非笼统的 Exception。
#   大厂规范：每个模块定义自己的异常前缀，如 ConfigError / ClientError / ToolError

class ConfigError(Exception):
    """配置缺失或格式错误时抛出

    使用场景：
        - ACTIVE_MODEL 指向了未注册的模型
        - 环境变量值不是合法数字
        - 必要的 API Key 缺失
    """
    pass


# ==================== 模型配置数据类 ====================
# 【知识点：frozen=True 的意义】
#   frozen=True 让 dataclass 实例变成"只读对象"：
#     cfg = ModelConfig(name="deepseek", ...)
#     cfg.name = "qwen"  # ❌ 抛出 FrozenInstanceError
#   这保证了配置一旦创建就不会被意外修改，类似 Java 中的 final 字段。

@dataclass(frozen=True)
class ModelConfig:
    """单个模型的连接配置（不可变数据对象）

    【设计思想】
    将"一个模型的所有连接信息"封装为一个对象，而非散落的变量。
    这是"值对象"（Value Object）模式：
        - 只关心"值是什么"，不关心"是谁的"
        - 两个 ModelConfig 字段完全相同 → 它们相等（__eq__ 自动生成）
        - frozen=True → 创建后不可修改 → 天然线程安全
    """
    name: str           # 别名，如 "deepseek"（用于日志和错误提示）
    provider: str       # API 格式："openai" | "anthropic"（决定用哪个客户端）
    api_key: str        # API 密钥（敏感信息，__repr__ 中脱敏显示）
    base_url: str       # API 基础地址（不同厂商地址不同）
    model: str          # 具体模型标识（如 "deepseek-chat", "qwen-plus"）
    # ---- 思考模式相关 ----
    # 各厂商对"思考模式"的实现完全不同，这里用两个字段抽象差异：
    supports_thinking: bool = False       # 该模型是否支持思考模式
    # thinking_param_type 决定"如何告诉 API 开启/关闭思考"：
    #   "extra_body"       → DeepSeek: extra_body={"thinking": {"type": "enabled"}}
    #   "reasoning_effort" → OpenAI o系列: reasoning_effort="high"
    #   "budget_tokens"    → Anthropic: thinking={"type": "enabled", "budget_tokens": N}
    #   "none"             → 不支持思考控制（Qwen / Kimi）
    thinking_param_type: str = "none"
    # ---- 参数约束相关 ----
    # 某些模型对请求参数有硬性限制（如 Kimi K2 系列只接受 temperature=1.0）。
    # fixed_params 中声明的参数将覆盖用户配置和全局配置，确保不会传入非法值。
    # 格式：{"参数名": 固定值}，如 {"temperature": 1.0}
    # 空字典表示无约束（大多数模型）。
    fixed_params: dict = field(default_factory=dict)

    def __repr__(self):
        """自定义打印格式（脱敏 API Key）

        【知识点：__repr__ vs __str__】
        - __repr__: 面向开发者，print(obj) 或调试时显示，应尽可能"可复现"
        - __str__: 面向用户，str(obj) 时显示，应"好看"
        - 如果只定义一个，优先定义 __repr__（__str__ 会回退到 __repr__）

        这里将 api_key 替换为 *** 防止日志泄露密钥。
        """
        return (
            f"ModelConfig(name={self.name!r}, provider={self.provider!r}, "
            f"model={self.model!r}, base_url={self.base_url!r}, "
            f"thinking={self.supports_thinking}, api_key=***)"
        )


# ==================== 配置主类 ====================
# 【知识点：类属性 vs 实例属性】
#   类属性（写在 class 下、方法外）：所有实例共享，如 KNOWN_MODELS
#   实例属性（写在 __init__ 中，self.xxx）：每个实例独有，如 self.APP_ENV
#
# 【知识点：@property 装饰器】
#   将方法伪装成属性访问：
#     cfg.active_model  而非  cfg.active_model()
#   好处：外部像读属性一样使用，内部可以做计算/校验/懒加载。
#   这是 Python 的"封装"方式（Java 用 getter/setter，Python 用 property）。
#
# 【知识点：@staticmethod 静态方法】
#   不接收 self 或 cls 参数，逻辑上属于类但不依赖类/实例状态。
#   本项目中 _get/_int/_float/_bool 是纯工具函数，不需要访问实例属性。

class Config:
    """
    生产级配置管理（多环境 + 多模型）— 全局唯一实例

    用法：
        from advanced_llm.config import cfg   # 导入全局单例

        # 获取当前激活模型的配置
        cfg.active_model          # → ModelConfig 对象
        cfg.active_model.api_key  # → API 密钥
        cfg.active_model.model    # → 模型标识（如 "deepseek-chat"）

        # 获取指定模型（不切换激活状态）
        cfg.get_model("qwen")     # → 千问的 ModelConfig

        # 通用配置
        cfg.DEBUG                 # → bool
        cfg.TIMEOUT               # → int（秒）
        cfg.MAX_TOKENS            # → int

    【单例模式实现方式】
    Python 中实现单例有多种方式：
        1. 模块级变量（本项目采用）：cfg = Config()，import 时只执行一次
        2. __new__ 方法拦截
        3. 装饰器 @singleton
        4. 元类 metaclass
    模块级变量最简单、最 Pythonic，因为 Python 模块本身就是单例的。
    """

    # 项目根目录（.env 文件所在目录 = advanced_llm/ 的上一级）
    # 【知识点：Path(__file__).resolve().parent.parent】
    #   __file__  → 当前文件路径（config.py 的绝对路径）
    #   .resolve() → 解析为绝对路径（消除 .. 和符号链接）
    #   .parent   → 上一级目录（advanced_llm/）
    #   .parent.parent → 再上一级（tutorial-advanced-llm/，即项目根目录）
    _BASE_DIR = Path(__file__).resolve().parent.parent

    # 已注册的模型别名列表（可扩展）
    KNOWN_MODELS = ["deepseek", "qwen", "kimi", "claude"]

    # provider 默认值映射（未在 .env.models 中显式指定时的回退）
    DEFAULT_PROVIDERS = {
        "deepseek": "openai",
        "qwen": "openai",
        "kimi": "openai",
        "claude": "anthropic",
    }

    # 各模型思考控制参数类型
    THINKING_PARAM_TYPES = {
        "deepseek": "extra_body",       # extra_body={"thinking": {"type": "enabled/disabled"}}
        "qwen": "none",                 # 千问暂无原生思考控制
        "kimi": "none",                 # Kimi 暂无原生思考控制
        "claude": "budget_tokens",      # thinking={"type": "enabled", "budget_tokens": N}
    }

    # 各模型是否支持思考模式
    THINKING_SUPPORT = {
        "deepseek": True,
        "qwen": False,
        "kimi": True,           # Kimi K2 系列默认开启思考
        "claude": True,
    }

    # ==================== 模型参数约束 ====================
    # 【设计思想：声明式参数约束】
    #   不同模型对请求参数有各自的硬性限制：
    #     - Kimi K2.7-code: temperature 只允许 1.0（思考模型，固定采样参数）
    #     - Kimi K2.6/K2.5: 思考模式 temperature=1.0，非思考模式 temperature=0.6
    #     - DeepSeek 思考模式: 不允许传 temperature
    #   通过 fixed_params 声明这些约束，客户端构建参数时自动应用，
    #   避免用户手动处理每个模型的差异。
    #
    # 【各模型参数约束详情（2026-07 最新）】
    #   kimi-k2.7-code / kimi-k2.7-code-highspeed:
    #     - temperature=1.0（固定，传其他值报错）
    #     - top_p=0.95（固定）
    #     - presence_penalty=0.0 / frequency_penalty=0.0（固定）
    #     - 思考默认开启，stream_options 仅支持 true
    #   kimi-k2.6 / kimi-k2.5:
    #     - 思考模式: temperature=1.0（固定）
    #     - 非思考模式: temperature=0.6（固定）
    #     - 传其他值报错，建议不显式设置
    #   deepseek-chat / deepseek-reasoner:
    #     - 思考模式下不支持 temperature / top_p
    #     - 非思考模式下正常支持
    MODEL_PARAM_CONSTRAINTS: dict[str, dict] = {
        "deepseek": {},     # 无固定约束（思考模式下由客户端逻辑自动跳过 temperature）
        "qwen": {},         # 千问无特殊约束
        "kimi": {           # Kimi K2 系列：采样参数固定
            "temperature": 1.0,
        },
        "claude": {},       # Claude 无特殊固定参数约束
    }

    def __init__(self):
        # 把所有的.env文件中的键值对加载到os.environ中，让后续代码可以通过os.getenv统一读取
        """
        ① .env              → 打底，提供所有变量的默认值
        ② 读 APP_ENV        → 确定当前环境（dev/prod/...）
        ③ .env.models       → 加载模型密钥
        ④ .env.{APP_ENV}    → 加载当前环境的专属配置
        ①③ 用 override=False：只填空位，已有的不覆盖
        ④ 用 override=True：强制覆盖同名变量
        """
        self._load_env_files()

        # ========== 环境标识 ==========
        """
        先加载 .env 打底，再探测环境名，然后按顺序加载 .env.models 和环境专属文件，后加载的覆盖先加载的，
        最终 os.environ 里每个变量只保留最高优先级的那个值。后续代码统一从 os.environ 读取，不再关心来源。
        """
        self.APP_ENV = self._get("APP_ENV", "dev")

        # ========== 思考模式 ==========
        self.ENABLE_THINKING = self._bool("ENABLE_THINKING", False)
        self.REASONING_EFFORT = self._get("REASONING_EFFORT", "high")
        self.THINKING_BUDGET_TOKENS = self._int("THINKING_BUDGET_TOKENS", 4096)

        # ========== 请求参数 ==========
        self.MAX_TOKENS = self._int("MAX_TOKENS", 4096)
        self.TEMPERATURE = self._float("TEMPERATURE", 0.7)
        self.TIMEOUT = self._int("TIMEOUT", 120)
        self.MAX_RETRIES = self._int("MAX_RETRIES", 3)

        # ========== 上下文管理 ==========
        self.MAX_CONTEXT_TOKENS = self._int("MAX_CONTEXT_TOKENS", 8000)
        self.MAX_HISTORY_ROUNDS = self._int("MAX_HISTORY_ROUNDS", 10)

        # ========== 应用 ==========
        self.DEBUG = self._bool("DEBUG", False)
        self.LOG_LEVEL = self._get("LOG_LEVEL", "INFO")

        # ========== 多模型注册 ==========
        self._models: dict[str, ModelConfig] = {}
        self._load_models()

        # ========== 当前激活模型 ==========
        active_name = self._get("ACTIVE_MODEL", "deepseek").lower()
        if active_name not in self._models:
            raise ConfigError(
                f"❌ ACTIVE_MODEL='{active_name}' 未注册\n"
                f"   可用模型: {list(self._models.keys())}\n"
                f"   请在 .env.models 中配置 {active_name.upper()}_API_KEY 等变量"
            )
        self._active_model_name = active_name

    # ==================== 环境文件加载 ====================

    def _load_env_files(self):
        """
        按优先级加载多个 .env 文件

        加载顺序（先加载的优先级低，后加载的覆盖前面的）：
            1. .env          — 通用默认值
            2. .env.models   — 模型密钥（独立于环境）
            3. .env.{APP_ENV} — 环境专属覆盖

        关键：override=False 保证系统环境变量始终最高优先级
        """
        # 先探测 APP_ENV（此时只从系统环境变量或 .env 中读取）
        load_dotenv(self._BASE_DIR / ".env", override=False)
        app_env = os.getenv("APP_ENV", "dev")

        # 加载模型配置
        models_file = self._BASE_DIR / ".env.models"
        if models_file.exists():
            load_dotenv(models_file, override=False)

        # 加载环境专属配置（覆盖 .env 中的同名变量）
        env_file = self._BASE_DIR / f".env.{app_env}"
        if env_file.exists():
            load_dotenv(env_file, override=True)

    # ==================== 多模型加载 ====================

    def _load_models(self):
        """
        从环境变量中加载所有已注册模型

        命名规则：{ALIAS}_API_KEY / {ALIAS}_BASE_URL / {ALIAS}_MODEL / {ALIAS}_PROVIDER
        例如：DEEPSEEK_API_KEY, QWEN_BASE_URL, KIMI_MODEL, CLAUDE_PROVIDER
        """
        for alias in self.KNOWN_MODELS:
            prefix = alias.upper()
            api_key = os.getenv(f"{prefix}_API_KEY", "")
            base_url = os.getenv(f"{prefix}_BASE_URL", "")
            model = os.getenv(f"{prefix}_MODEL", "")
            provider = os.getenv(
                f"{prefix}_PROVIDER",
                self.DEFAULT_PROVIDERS.get(alias, "openai")
            ).lower()

            # 只注册有 API_KEY 的模型（允许部分模型未配置）
            if api_key:
                self._models[alias] = ModelConfig(
                    name=alias,
                    provider=provider,
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                    supports_thinking=self.THINKING_SUPPORT.get(alias, False),
                    thinking_param_type=self.THINKING_PARAM_TYPES.get(alias, "none"),
                    fixed_params=self.MODEL_PARAM_CONSTRAINTS.get(alias, {}),
                )

    # ==================== 公开接口 ====================

    @property
    def active_model(self) -> ModelConfig:
        """当前激活的模型配置"""
        return self._models[self._active_model_name]

    def get_model(self, name: str) -> ModelConfig:
        """获取指定模型配置（不切换激活状态）"""
        name = name.lower()
        if name not in self._models:
            available = list(self._models.keys())
            raise ConfigError(
                f"❌ 模型 '{name}' 未配置\n"
                f"   可用模型: {available}\n"
                f"   请在 .env.models 中添加 {name.upper()}_API_KEY 等变量"
            )
        return self._models[name]

    @property
    def available_models(self) -> list[str]:
        """所有已配置的模型别名"""
        return list(self._models.keys())

    # ==================== 内部工具方法 ====================
    # 【知识点：@staticmethod 静态方法】
    #   静态方法不需要 self 参数，不能访问实例属性。
    #   适合放"纯工具函数"：输入 → 输出，不依赖对象状态。
    #   放在类内部是为了逻辑归组（这些函数都是为 Config 服务的）。
    #
    # 【知识点：os.getenv(key, default)】
    #   从系统环境变量中读取值。如果不存在，返回 default。
    #   所有值都是 str 类型，需要手动转换为 int/float/bool。
    #   这就是为什么需要 _int/_float/_bool 辅助方法。

    @staticmethod
    def _get(key: str, default: str = "") -> str:
        """读取字符串类型环境变量"""
        return os.getenv(key, default)

    @staticmethod
    def _int(key: str, default: int = 0) -> int:
        """读取整数类型环境变量（带格式校验）

        【知识点：异常处理的最佳实践】
        不要静默忽略错误！配置错误应该在启动时就暴露（Fail Fast 原则），
        而不是运行到一半才因为类型错误崩溃。
        """
        value = os.getenv(key)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            raise ConfigError(f"❌ 环境变量 {key}='{value}' 不是合法整数")

    @staticmethod
    def _float(key: str, default: float = 0.0) -> float:
        """读取浮点数类型环境变量（带格式校验）"""
        value = os.getenv(key)
        if value is None:
            return default
        try:
            return float(value)
        except ValueError:
            raise ConfigError(f"❌ 环境变量 {key}='{value}' 不是合法浮点数")

    @staticmethod
    def _bool(key: str, default: bool = False) -> bool:
        """读取布尔类型环境变量

        【知识点：为什么不能直接 bool(os.getenv(key))？】
        因为 bool("false") == True！（非空字符串都是 True）
        所以必须手动判断字符串内容："true"/"1"/"yes"/"on" → True，其余 → False
        """
        value = os.getenv(key)
        if value is None:
            return default
        return value.strip().lower() in ("true", "1", "yes", "on")

    def __repr__(self):
        lines = [
            f"【当前配置】环境={self.APP_ENV} | 激活模型={self._active_model_name}",
            f"  可用模型: {self.available_models}",
            f"  思考模式: {self.ENABLE_THINKING}",
            f"  MAX_TOKENS: {self.MAX_TOKENS}",
            f"  TEMPERATURE: {self.TEMPERATURE}",
            f"  MAX_CONTEXT_TOKENS: {self.MAX_CONTEXT_TOKENS}",
        ]
        return "\n".join(lines)


# ==================== 全局单例 ====================
# 【知识点：模块级单例】
#   Python 的 import 机制保证：同一个模块只会被执行一次。
#   因此 `cfg = Config()` 无论被 import 多少次，Config() 只执行一次。
#   所有 `from advanced_llm.config import cfg` 拿到的都是同一个对象。
#
#   验证方式：
#     from advanced_llm.config import cfg as cfg1
#     from advanced_llm.config import cfg as cfg2
#     print(cfg1 is cfg2)  # → True（同一个对象）
cfg = Config()
