import os

import requests
from dotenv import load_dotenv

ALIYUN_RERANK_URL = "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"
ALIYUN_RERANK_MODEL = "qwen3-rerank"


def aliyun_rerank(query: str, candidates: list[str], top_n: int = 5) -> list[tuple[int, float]]:
    if not candidates:
        return []

    load_dotenv()
    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    if not api_key:
        raise ValueError("DASHSCOPE_API_KEY is not set. Please configure it in ThinkRAG/.env.")

    payload = {
        "model": ALIYUN_RERANK_MODEL,
        "query": query,
        "documents": candidates,
        "top_n": min(int(top_n), len(candidates)),
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    resp = requests.post(ALIYUN_RERANK_URL, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()

    results = resp.json().get("results", [])
    return [(int(item["index"]), float(item["relevance_score"])) for item in results]
