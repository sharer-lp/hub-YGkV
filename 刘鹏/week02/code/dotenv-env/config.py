"""
生产级多环境 + 多模型配置管理

==================== 架构设计 ====================

文件结构：
    dotenv-env/
    ├── .env              ← 通用默认配置（所有环境共享）
    ├── .env.dev          ← 开发环境覆盖
    ├── .env.prod         ← 生产环境覆盖
    ├── .env.models       ← 多模型密钥与连接（独立管理）
    ├── .env.example      ← 模板（提交到 Git）
    └── config.py         ← 本文件

加载优先级（后加载的覆盖先加载的）：
    系统环境变量 > .env.{APP_ENV} > .env.models > .env

环境切换：
    方式1：系统环境变量  APP_ENV=prod python main.py
    方式2：PowerShell    $env:APP_ENV="prod"; python main.py
    方式3：.env 中写     APP_ENV=prod（不推荐，失去了环境隔离意义）

多模型切换：
    .env.models 中修改 ACTIVE_MODEL=deepseek / qwen / openai
    或系统环境变量 ACTIVE_MODEL=qwen python main.py

==================================================
"""

import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv


# ==================== 异常 ====================

class ConfigError(Exception):
    """配置缺失或格式错误时抛出"""
    pass


# ==================== 模型配置数据类 ====================

@dataclass
class ModelConfig:
    """单个模型的连接配置"""
    name: str           # 别名，如 "deepseek"
    api_key: str
    base_url: str
    model: str

    def __repr__(self):
        return f"ModelConfig(name={self.name!r}, model={self.model!r}, base_url={self.base_url!r}, api_key=***)"


# ==================== 配置主类 ====================

class Config:
    """
    生产级配置管理（多环境 + 多模型）

    用法：
        from config import cfg

        # 获取当前激活模型
        cfg.active_model          # ModelConfig 对象
        cfg.active_model.api_key

        # 获取指定模型
        cfg.get_model("qwen")     # 切换到千问

        # 通用配置
        cfg.DEBUG
        cfg.TIMEOUT
    """

    # 项目根目录（config.py 所在目录）
    _BASE_DIR = Path(__file__).resolve().parent

    # 已注册的模型别名列表（可扩展）
    KNOWN_MODELS = ["deepseek", "qwen", "openai"]

    def __init__(self):
        self._load_env_files()

        # ========== 环境标识 ==========
        self.APP_ENV = self._get("APP_ENV", "dev")

        # ========== 思考模式 ==========
        self.ENABLE_THINKING = self._bool("ENABLE_THINKING", False)
        self.REASONING_EFFORT = self._get("REASONING_EFFORT", "high")

        # ========== 请求参数 ==========
        self.MAX_TOKENS = self._int("MAX_TOKENS", 4096)
        self.TIMEOUT = self._int("TIMEOUT", 120)
        self.MAX_RETRIES = self._int("MAX_RETRIES", 3)

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
            load_dotenv(env_file, override=True)  # 环境文件覆盖通用 .env

    # ==================== 多模型加载 ====================

    def _load_models(self):
        """
        从环境变量中加载所有已注册模型

        命名规则：{ALIAS}_API_KEY / {ALIAS}_BASE_URL / {ALIAS}_MODEL
        例如：DEEPSEEK_API_KEY, QWEN_BASE_URL, OPENAI_MODEL
        """
        for alias in self.KNOWN_MODELS:
            prefix = alias.upper()
            api_key = os.getenv(f"{prefix}_API_KEY", "")
            base_url = os.getenv(f"{prefix}_BASE_URL", "")
            model = os.getenv(f"{prefix}_MODEL", "")

            # 只注册有 API_KEY 的模型（允许部分模型未配置）
            if api_key:
                self._models[alias] = ModelConfig(
                    name=alias,
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                )

    # ==================== 公开接口 ====================

    @property
    def active_model(self) -> ModelConfig:
        """当前激活的模型配置"""
        return self._models[self._active_model_name]

    def get_model(self, name: str) -> ModelConfig:
        """
        获取指定模型配置（不切换激活状态）

        用法：cfg.get_model("qwen")
        """
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

    # ==================== 向后兼容（旧代码不用改） ====================

    @property
    def DEEPSEEK_API_KEY(self) -> str:
        return self.active_model.api_key

    @property
    def DEEPSEEK_BASE_URL(self) -> str:
        return self.active_model.base_url

    @property
    def DEEPSEEK_MODEL(self) -> str:
        return self.active_model.model

    # ==================== 内部工具方法 ====================

    @staticmethod
    def _require(key: str) -> str:
        value = os.getenv(key)
        if not value:
            raise ConfigError(
                f"❌ 缺少必填环境变量: {key}\n"
                f"   请在 .env 或 .env.models 中添加: {key}=xxx"
            )
        return value

    @staticmethod
    def _get(key: str, default: str = "") -> str:
        return os.getenv(key, default)

    @staticmethod
    def _int(key: str, default: int = 0) -> int:
        value = os.getenv(key)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            raise ConfigError(f"❌ 环境变量 {key}='{value}' 不是合法整数")

    @staticmethod
    def _bool(key: str, default: bool = False) -> bool:
        value = os.getenv(key)
        if value is None:
            return default
        return value.strip().lower() in ("true", "1", "yes", "on")

    def __repr__(self):
        sensitive_keys = {"KEY", "SECRET", "PASSWORD", "TOKEN"}
        lines = [
            f"【当前配置】环境={self.APP_ENV} | 激活模型={self._active_model_name}",
            f"  可用模型: {self.available_models}",
        ]
        for k, v in sorted(self.__dict__.items()):
            if k.startswith("_"):
                continue
            if any(s in k.upper() for s in sensitive_keys):
                v = "***已脱敏***" if v else "(空)"
            lines.append(f"  {k} = {v}")
        return "\n".join(lines)


# ==================== 全局单例 ====================
cfg = Config()
