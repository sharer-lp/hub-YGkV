"""
项目全局配置

所有文件路径基于项目根目录的绝对路径，不依赖运行时工作目录。
"""
import os

# 项目根目录（01-intent-classify/）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ==================== 正则规则配置 ====================
REGEX_RULE = {
    "FilmTele-Play": ["播放", "电视剧"],  # 句子是不是包含特定的单词，做出分类
    "HomeAppliance-Control": ["空调", "广播"]
}

# ==================== 分类类别 ====================
CATEGORY_NAME = [
    'Travel-Query', 'Music-Play', 'FilmTele-Play', 'Video-Play',
    'Radio-Listen', 'HomeAppliance-Control', 'Weather-Query',
    'Alarm-Update', 'Calendar-Query', 'TVProgram-Play', 'Audio-Play',
    'Other'
]

# ==================== 模型文件路径（绝对路径） ====================
TFIDF_MODEL_PKL_PATH = os.path.join(BASE_DIR, "assets", "weights", "tfidf_ml.pkl")

BERT_MODEL_PKL_PATH = os.path.join(BASE_DIR, "assets", "weights", "bert.pt")
BERT_MODEL_PERTRAINED_PATH = os.path.join(BASE_DIR, "assets", "models", "bert-base-chinese")

# ==================== 数据集路径 ====================
DATASET_CSV_PATH = os.path.join(BASE_DIR, "assets", "dataset", "dataset.csv")
STOPWORDS_PATH = os.path.join(BASE_DIR, "assets", "dataset", "baidu_stopwords.txt")

# ==================== LLM 配置 ====================
LLM_OPENAI_SERVER_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_OPENAI_API_KEY = "sk-3b63e3a86139434e94dc5e64eee50745"
LLM_MODEL_NAME = "qwen-plus"
