import pytest

from src.rag.reranker import ALIYUN_RERANK_URL, aliyun_rerank


class FakeResponse:
    def __init__(self, data=None, error=None):
        self.data = data or {}
        self.error = error

    def raise_for_status(self):
        if self.error:
            raise self.error

    def json(self):
        return self.data


def test_aliyun_rerank_posts_expected_payload(monkeypatch):
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured.update({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return FakeResponse(
            {
                "results": [
                    {"index": 1, "relevance_score": 0.91},
                    {"index": 0, "relevance_score": 0.42},
                ]
            }
        )

    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setattr("src.rag.reranker.requests.post", fake_post)

    result = aliyun_rerank("query", ["doc-a", "doc-b"], top_n=5)

    assert captured["url"] == ALIYUN_RERANK_URL
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["json"]["model"] == "qwen3-rerank"
    assert captured["json"]["top_n"] == 2
    assert result == [(1, 0.91), (0, 0.42)]


def test_aliyun_rerank_raises_http_errors(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setattr(
        "src.rag.reranker.requests.post",
        lambda *args, **kwargs: FakeResponse(error=RuntimeError("bad status")),
    )

    with pytest.raises(RuntimeError):
        aliyun_rerank("query", ["doc-a"])


def test_aliyun_rerank_requires_api_key(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

    with pytest.raises(ValueError):
        aliyun_rerank("query", ["doc-a"])
