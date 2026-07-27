from typing import Union

from fastapi import FastAPI

app = FastAPI()

# fastapi 将 http的网络请求 转换为本地函数的调用

# http://0.0.0.0:8000/ -> read_root
@app.get("/")
def read_root():
    return {"Hello": "World"}

# http://0.0.0.0:8000/items/{item_id} -> read_item
@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}