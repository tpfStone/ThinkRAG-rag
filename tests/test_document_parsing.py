from types import SimpleNamespace

from llama_index.core.schema import TextNode

from server.document_parsing import (
    DashScopeParseProvider,
    ParsedFileResult,
    NativeParseProvider,
    parse_input_files,
    sha256_file,
)


def doc(text, file_name="doc.pdf", page="1"):
    return SimpleNamespace(text=text, metadata={"file_name": file_name, "page_label": page})


class FakeNativeProvider(NativeParseProvider):
    def __init__(self, documents):
        self.documents = documents

    def parse_file(self, file_path, file_hash, chunk_size, chunk_overlap, extra_metadata=None):
        for document in self.documents:
            document.metadata["file_hash"] = file_hash
            document.metadata["parse_provider"] = "native"
            document.metadata["parse_status"] = "success"
        return ParsedFileResult(
            file_path=file_path,
            file_hash=file_hash,
            native_documents=self.documents,
            quality_documents=self.documents,
            parse_file_report={
                "file_name": file_path.name,
                "provider": "native",
                "source": "native",
                "status": "success",
                "candidate_pages": 0,
                "repaired_pages": 0,
                "failed_pages": 0,
            },
        )


class FakeDashScopeProvider:
    def __init__(self, nodes=None):
        self.nodes = nodes or []
        self.calls = 0

    def parse_file(self, file_path, file_hash, chunk_size, chunk_overlap, extra_metadata=None):
        self.calls += 1
        return ParsedFileResult(
            file_path=file_path,
            file_hash=file_hash,
            api_nodes=self.nodes,
            parse_file_report={
                "file_name": file_path.name,
                "provider": "dashscope_parse",
                "source": "dashscope_parse",
                "status": "success" if self.nodes else "failed",
                "candidate_pages": 0,
                "repaired_pages": 0,
                "failed_pages": 0,
            },
        )


def test_auto_provider_keeps_high_quality_native_pdf(tmp_path):
    file_path = tmp_path / "native.pdf"
    file_path.write_bytes(b"fake")
    native_docs = [doc("This page has enough readable native text. " * 3, "native.pdf")]
    dashscope = FakeDashScopeProvider(nodes=[TextNode(text="should not be used")])

    result = parse_input_files(
        [file_path],
        512,
        64,
        parse_provider="auto",
        native_provider=FakeNativeProvider(native_docs),
        dashscope_provider=dashscope,
    )

    assert result.native_documents == native_docs
    assert result.api_nodes == []
    assert dashscope.calls == 0
    assert result.quality_report["files"][0]["parse_source"] == "native"


def test_auto_provider_routes_empty_pdf_to_dashscope(tmp_path):
    file_path = tmp_path / "scan.pdf"
    file_path.write_bytes(b"fake")
    native_docs = [doc("", "scan.pdf", "1"), doc("", "scan.pdf", "2")]
    api_nodes = [TextNode(text="Parsed text from API", metadata={"file_name": "scan.pdf"})]
    dashscope = FakeDashScopeProvider(nodes=api_nodes)

    result = parse_input_files(
        [file_path],
        512,
        64,
        parse_provider="auto",
        native_provider=FakeNativeProvider(native_docs),
        dashscope_provider=dashscope,
    )

    assert result.native_documents == []
    assert result.api_nodes == api_nodes
    assert dashscope.calls == 1
    assert result.quality_report["summary"]["parse_repaired_pages"] == 2
    assert result.quality_report["files"][0]["parse_source"] == "dashscope_parse"


def test_dashscope_provider_uses_node_cache_without_reparse(tmp_path, monkeypatch):
    file_path = tmp_path / "scan.pdf"
    file_path.write_bytes(b"fake")
    file_hash = sha256_file(file_path)
    provider = DashScopeParseProvider(cache_dir=tmp_path / "cache", api_key="test-key")
    calls = {"parse": 0}

    def fake_call_dashscope(path):
        calls["parse"] += 1
        return [{"DASHSCOPE_DOCMIND": "{\"pages\": []}", "job_id": "job-1"}]

    monkeypatch.setattr(provider, "_call_dashscope", fake_call_dashscope)
    monkeypatch.setattr(
        provider,
        "_nodes_from_documents",
        lambda documents, chunk_size, chunk_overlap: [TextNode(text="cached text", metadata={"file_name": "scan.pdf"})],
    )

    first = provider.parse_file(file_path, file_hash, 512, 64)
    second = provider.parse_file(file_path, file_hash, 512, 64)

    assert calls["parse"] == 1
    assert first.api_nodes[0].text == "cached text"
    assert second.api_nodes[0].text == "cached text"
    assert second.parse_file_report["source"] == "cache"


def test_dashscope_provider_reports_missing_api_key(tmp_path):
    file_path = tmp_path / "scan.pdf"
    file_path.write_bytes(b"fake")
    file_hash = sha256_file(file_path)
    provider = DashScopeParseProvider(cache_dir=tmp_path / "cache", api_key="")

    result = provider.parse_file(file_path, file_hash, 512, 64)

    assert result.api_nodes == []
    assert result.parse_file_report["status"] == "failed"
    assert "DASHSCOPE_API_KEY" in result.parse_file_report["error"]
