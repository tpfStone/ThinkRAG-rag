from types import SimpleNamespace

import server.retriever as retriever_module
from llama_index.core.llms import MockLLM
from server.retriever import SimpleFusionRetriever


def test_simple_fusion_retriever_disables_llm_query_generation(monkeypatch):
    captured = {}

    class FakeVectorRetriever:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeBM25Retriever:
        @classmethod
        def from_defaults(cls, **kwargs):
            return cls()

    def fake_query_fusion_init(self, retrievers, **kwargs):
        captured["retrievers"] = retrievers
        captured["kwargs"] = kwargs

    monkeypatch.setattr(retriever_module, "VectorIndexRetriever", FakeVectorRetriever)
    monkeypatch.setattr(retriever_module, "SimpleBM25Retriever", FakeBM25Retriever)
    monkeypatch.setattr(retriever_module.QueryFusionRetriever, "__init__", fake_query_fusion_init)

    index = SimpleNamespace(docstore=SimpleNamespace(docs={"n1": object(), "n2": object()}))

    SimpleFusionRetriever(vector_index=index, top_k=2)

    assert captured["kwargs"]["num_queries"] == 1
    assert isinstance(captured["kwargs"]["llm"], MockLLM)
