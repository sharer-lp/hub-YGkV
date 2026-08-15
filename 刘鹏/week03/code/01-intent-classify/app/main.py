"""
FastAPI 应用入口

启动方式（在项目根目录 01-intent-classify/ 下执行）：
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

设计说明：
    模型采用懒加载策略，服务启动时不加载任何模型文件，
    仅在首次调用对应接口时才加载，避免模型文件缺失导致服务无法启动。
"""
# python自带库
import time
import traceback
from functools import lru_cache

# 第三方库
from fastapi import FastAPI

# 自己写的模块（仅导入不依赖模型文件的模块）
from app.data_schema import TextClassifyResponse, TextClassifyRequest
from app.model.regex_rule import model_for_regex
from app.logger import logger

app = FastAPI(title="意图分类服务", version="1.0.0")


# ==================== 懒加载模型（首次调用时加载，之后缓存） ====================

@lru_cache(maxsize=1)
def _get_tfidf_model():
    """懒加载 TF-IDF 模型"""
    from app.model.tfidf_ml import model_for_tfidf
    logger.info("TF-IDF 模型加载完成")
    return model_for_tfidf


@lru_cache(maxsize=1)
def _get_bert_model():
    """懒加载 BERT 模型"""
    from app.model.bert import model_for_bert
    logger.info("BERT 模型加载完成")
    return model_for_bert


@lru_cache(maxsize=1)
def _get_gpt_model():
    """懒加载 LLM (GPT) 模型"""
    from app.model.prompt import model_for_gpt
    logger.info("LLM 模型加载完成")
    return model_for_gpt


@app.post("/v1/text-cls/regex")
def regex_classify(req: TextClassifyRequest) -> TextClassifyResponse:
    """
    利用正则表达式进行文本分类

    :param req: 请求体
    """
    start_time = time.time()
    response = TextClassifyResponse(
        request_id=req.request_id,
        request_text=req.request_text,
        classify_result="",
        classify_time=0,
        error_msg=""
    )

    logger.info(f"{req.request_id} {req.request_text}")  # 打印请求
    try:
        response.classify_result = model_for_regex(req.request_text)
        response.error_msg = "ok"
    except Exception as err:
        response.classify_result = ""
        response.error_msg = traceback.format_exc()

    response.classify_time = round(time.time() - start_time, 3)
    return response


@app.post("/v1/text-cls/tfidf")
def tfidf_classify(req: TextClassifyRequest) -> TextClassifyResponse:
    """
    利用TFIDF进行文本分类

    :param req: 请求体
    """
    start_time = time.time()
    response = TextClassifyResponse(
        request_id=req.request_id,
        request_text=req.request_text,
        classify_result="",
        classify_time=0,
        error_msg=""
    )
    logger.info(f"Get request: {req.model_dump_json()}")

    try:
        model_for_tfidf = _get_tfidf_model()
        response.classify_result = model_for_tfidf(req.request_text)
        response.error_msg = "ok"
    except Exception as err:
        response.classify_result = ""
        response.error_msg = traceback.format_exc()

    response.classify_time = round(time.time() - start_time, 3)
    return response


@app.post("/v1/text-cls/bert")
def bert_classify(req: TextClassifyRequest) -> TextClassifyResponse:
    """
    利用BERT进行文本分类

    :param req: 请求体
    """
    start_time = time.time()

    response = TextClassifyResponse(
        request_id=req.request_id,
        request_text=req.request_text,
        classify_result="",
        classify_time=0,
        error_msg=""
    )
    try:
        model_for_bert = _get_bert_model()
        response.classify_result = model_for_bert(req.request_text)
        response.error_msg = "ok"
    except Exception as err:
        response.classify_result = ""
        response.error_msg = traceback.format_exc()

    response.classify_time = round(time.time() - start_time, 3)
    return response


@app.post("/v1/text-cls/gpt")
def gpt_classify(req: TextClassifyRequest) -> TextClassifyResponse:
    """
    利用大语言模型进行文本分类

    :param req: 请求体
    """
    start_time = time.time()
    response = TextClassifyResponse(
        request_id=req.request_id,
        request_text=req.request_text,
        classify_result="",
        classify_time=0,
        error_msg=""
    )

    try:
        model_for_gpt = _get_gpt_model()
        response.classify_result = model_for_gpt(req.request_text)
        response.error_msg = "ok"
    except Exception as err:
        response.classify_result = ""
        response.error_msg = traceback.format_exc()

    response.classify_time = round(time.time() - start_time, 3)
    return response
