"""
日志配置模块

日志文件输出到项目根目录下的 app.log，同时输出到控制台。
"""
import logging
import os

from app.config import BASE_DIR

LOG_PATH = os.path.join(BASE_DIR, "app.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),  # 输出到文件
        logging.StreamHandler(),                           # 同时输出到控制台
    ]
)

logger = logging.getLogger(__name__)
