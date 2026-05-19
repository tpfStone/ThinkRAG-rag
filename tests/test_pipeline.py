from types import SimpleNamespace

import src.rag.pipeline as pipeline_module
from src.rag.pipeline import REFUSAL_ANSWER, RAGPipeline


class FakeCompletions:
    def __init__(self, content):
        self.content = content
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeClient:
    def __init__(self, content="fake answer"):
        self.chat = SimpleNamespace(completions=FakeCompletions(content))


class FakeInnerNode:
    def __init__(self, text, metadata, node_id):
        self.text = text
        self.metadata = metadata
        self.node_id = node_id


class FakeNodeWithScore:
    def __init__(self, text, score, file_name, node_id):
        self.score = score
        self.node = FakeInnerNode(text, {"file_name": file_name, "page_label": "1"}, node_id)


class FakeRetriever:
    nodes = []

    def __init__(self, vector_index, top_k):
        self.vector_index = vector_index
        self.top_k = top_k

    def retrieve(self, question):
        return self.nodes[: self.top_k]


def test_pipeline_without_retrieval_calls_llm_only():
    config = {"retrieval": {"enabled": False}}
    client = FakeClient("direct answer")

    result = RAGPipeline(config, client=client).answer("question")

    assert result["answer"] == "direct answer"
    assert result["sources"] == []
    assert client.chat.completions.calls[0]["model"] == "qwen-plus"


def test_pipeline_refuses_low_score_results(monkeypatch):
    FakeRetriever.nodes = [
        FakeNodeWithScore("doc one", 0.3, "a.pdf", "n1"),
        FakeNodeWithScore("doc two", 0.2, "b.pdf", "n2"),
    ]
    monkeypatch.setattr(pipeline_module, "SimpleFusionRetriever", FakeRetriever)
    config = {
        "retrieval": {"enabled": True, "top_k": 2, "use_reranker": False},
        "refuse_gate": {"enabled": True, "max_threshold": 0.5, "spread_threshold": 0.05},
    }
    client = FakeClient("should not be used")

    result = RAGPipeline(config, index=object(), client=client).answer("question")

    assert result["refused"] is True
    assert result["answer"] == REFUSAL_ANSWER
    assert result["refusal_reason"].startswith("max_score_too_low")
    assert client.chat.completions.calls == []


def test_pipeline_refuses_when_retrieved_chunks_are_empty(monkeypatch):
    FakeRetriever.nodes = [
        FakeNodeWithScore("", 0.9, "empty.pdf", "n1"),
        FakeNodeWithScore("   ", 0.8, "empty.pdf", "n2"),
    ]
    monkeypatch.setattr(pipeline_module, "SimpleFusionRetriever", FakeRetriever)
    config = {
        "retrieval": {"enabled": True, "top_k": 2, "use_reranker": False},
        "refuse_gate": {"enabled": True, "max_threshold": 0.5, "spread_threshold": 0.05},
    }
    client = FakeClient("should not be used")

    result = RAGPipeline(config, index=object(), client=client).answer("question")

    assert result["refused"] is True
    assert result["refusal_reason"] == "retrieved_chunks_empty"
    assert result["answer"] == REFUSAL_ANSWER
    assert result["warnings"][0].startswith("retrieved_chunks_empty:")
    assert client.chat.completions.calls == []


def test_pipeline_skips_empty_chunks_and_answers_with_valid_chunks(monkeypatch):
    FakeRetriever.nodes = [
        FakeNodeWithScore("", 0.95, "empty.pdf", "n1"),
        FakeNodeWithScore("valid evidence", 0.8, "valid.pdf", "n2"),
    ]
    monkeypatch.setattr(pipeline_module, "SimpleFusionRetriever", FakeRetriever)
    config = {
        "retrieval": {"enabled": True, "top_k": 2, "use_reranker": False},
        "refuse_gate": {"enabled": True, "max_threshold": 0.5, "spread_threshold": 0.0},
    }

    result = RAGPipeline(config, index=object(), client=FakeClient("grounded answer")).answer("question")

    assert result["refused"] is False
    assert result["answer"] == "grounded answer"
    assert result["sources"] == ["valid.pdf"]
    assert result["max_score"] == 0.8
    assert result["warnings"][0].startswith("retrieved_chunks_empty:")


def test_pipeline_reranks_and_runs_consistency_check(monkeypatch):
    FakeRetriever.nodes = [
        FakeNodeWithScore("first doc", 0.4, "a.pdf", "n1"),
        FakeNodeWithScore("second doc", 0.8, "b.pdf", "n2"),
    ]
    monkeypatch.setattr(pipeline_module, "SimpleFusionRetriever", FakeRetriever)
    monkeypatch.setattr(pipeline_module, "aliyun_rerank", lambda question, texts, top_n: [(1, 0.92)])
    monkeypatch.setattr(pipeline_module, "verify_consistency", lambda answer, evidence, client: ("Y", "ok"))
    config = {
        "retrieval": {"enabled": True, "top_k": 1, "use_reranker": True, "initial_top_k": 2},
        "prompt": {"strict_mode": True},
        "consistency_check": {"enabled": True},
    }

    result = RAGPipeline(config, index=object(), client=FakeClient("grounded answer")).answer("question")

    assert result["answer"] == "grounded answer"
    assert result["sources"] == ["b.pdf"]
    assert result["max_score"] == 0.92
    assert result["consistency_label"] == "Y"
    assert result["source_details"][0]["file"] == "b.pdf"


def test_pipeline_continues_when_reranker_fails(monkeypatch):
    FakeRetriever.nodes = [
        FakeNodeWithScore("first doc", 0.7, "a.pdf", "n1"),
        FakeNodeWithScore("second doc", 0.6, "b.pdf", "n2"),
    ]
    monkeypatch.setattr(pipeline_module, "SimpleFusionRetriever", FakeRetriever)

    def fail_rerank(question, texts, top_n):
        raise RuntimeError("rerank unavailable")

    monkeypatch.setattr(pipeline_module, "aliyun_rerank", fail_rerank)
    config = {
        "retrieval": {"enabled": True, "top_k": 1, "use_reranker": True, "initial_top_k": 2},
    }

    result = RAGPipeline(config, index=object(), client=FakeClient("fallback answer")).answer("question")

    assert result["answer"] == "fallback answer"
    assert result["sources"] == ["a.pdf"]
    assert result["max_score"] == 0.7
    assert result["warnings"][0].startswith("reranker_failed:")


def test_pipeline_continues_when_consistency_check_fails(monkeypatch):
    FakeRetriever.nodes = [
        FakeNodeWithScore("first doc", 0.7, "a.pdf", "n1"),
    ]
    monkeypatch.setattr(pipeline_module, "SimpleFusionRetriever", FakeRetriever)

    def fail_consistency(answer, evidence, client):
        raise RuntimeError("consistency unavailable")

    monkeypatch.setattr(pipeline_module, "verify_consistency", fail_consistency)
    config = {
        "retrieval": {"enabled": True, "top_k": 1, "use_reranker": False},
        "consistency_check": {"enabled": True},
    }

    result = RAGPipeline(config, index=object(), client=FakeClient("grounded answer")).answer("question")

    assert result["answer"] == "grounded answer"
    assert result["consistency_label"] == "N/A"
    assert result["consistency_reason"] == "Consistency check skipped."
    assert result["warnings"][0].startswith("consistency_check_failed:")
