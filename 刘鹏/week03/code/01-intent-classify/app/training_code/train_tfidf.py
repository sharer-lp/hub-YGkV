"""
TF-IDF + LinearSVC 模型训练脚本

运行方式（在项目根目录 01-intent-classify/ 下执行）：
    python -m app.training_code.train_tfidf
"""
import pandas as pd
import jieba
from joblib import dump
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

from app.config import DATASET_CSV_PATH, STOPWORDS_PATH, TFIDF_MODEL_PKL_PATH

train_data = pd.read_csv(DATASET_CSV_PATH, sep='\t', header=None)

cn_stopwords = pd.read_csv(STOPWORDS_PATH, header=None)[0].values

train_data[0] = train_data[0].apply(lambda x: " ".join([w for w in jieba.lcut(x) if w not in cn_stopwords]))

tfidf = TfidfVectorizer(ngram_range=(1, 1))

train_tfidf = tfidf.fit_transform(train_data[0])

model = LinearSVC()
model.fit(train_tfidf, train_data[1])

# 模型保存
dump((tfidf, model), TFIDF_MODEL_PKL_PATH)
print(f"模型已保存至: {TFIDF_MODEL_PKL_PATH}")
