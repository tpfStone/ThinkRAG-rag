import os

from openai import OpenAI

ALIYUN_COMPATIBLE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def get_aliyun_client() -> OpenAI:
    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    if not api_key:
        raise ValueError("DASHSCOPE_API_KEY is not set. Please configure it in ThinkRAG/.env.")
    return OpenAI(api_key=api_key, base_url=ALIYUN_COMPATIBLE_BASE_URL)
