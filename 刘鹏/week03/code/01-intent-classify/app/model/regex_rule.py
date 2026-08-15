"""
正则规则分类模型

基于关键词匹配进行意图分类，无需训练，开箱即用。
"""
import re
from typing import Union, List

from app.config import REGEX_RULE

# 预编译正则表达式，提升匹配性能
REGEX_RULE_COMPILED = {}
for category in REGEX_RULE.keys():
    REGEX_RULE_COMPILED[category] = re.compile("|".join(REGEX_RULE[category]))


def model_for_regex(request_text: Union[str, List[str]]) -> Union[str, List[str]]:
    classify_result: Union[str, List[str]] = []

    if isinstance(request_text, str):
        for category in REGEX_RULE_COMPILED.keys():
            if REGEX_RULE_COMPILED[category].findall(request_text):
                classify_result.append(category)
        if not classify_result:
            classify_result.append("Other")
    elif isinstance(request_text, list):
        classify_result = []
        for text in request_text:
            is_classified = False
            for category in REGEX_RULE_COMPILED.keys():
                if REGEX_RULE_COMPILED[category].findall(text):
                    classify_result.append(category)
                    is_classified = True

            if not is_classified:
                classify_result.append("Other")
    else:
        raise Exception("格式不支持")

    return classify_result
