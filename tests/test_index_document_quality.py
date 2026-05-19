from types import SimpleNamespace

import server.index as index_module
from server.document_parsing import ParsedFilesResult, attach_parse_report
from server.document_quality import build_quality_report
from server.index import IndexManager


def doc(text, file_name="doc.pdf", page="1"):
    return SimpleNamespace(text=text, metadata={"file_name": file_name, "page_label": page})


class FakePipeline:
    seen_documents = None
    seen_nodes = None

    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs

    def run(self, documents=None, nodes=None):
        self.__class__.seen_documents = documents
        self.__class__.seen_nodes = nodes
        return ["node-1"]


def make_manager():
    manager = IndexManager.__new__(IndexManager)
    manager.inserted_nodes = []

    def insert_nodes(nodes):
        manager.inserted_nodes.append(nodes)
        return object()

    manager.insert_nodes = insert_nodes
    return manager


def parsed_result(native_documents=None, api_nodes=None, quality_documents=None, parse_report=None):
    quality_documents = quality_documents if quality_documents is not None else (native_documents or [])
    quality_report = attach_parse_report(build_quality_report(quality_documents), parse_report or {"files": []})
    return ParsedFilesResult(
        native_documents=native_documents or [],
        api_nodes=api_nodes or [],
        quality_report=quality_report,
        parse_report=parse_report or {"files": []},
    )


def test_load_files_does_not_ingest_all_empty_documents(monkeypatch):
    documents = [doc("", "empty.pdf", "1")]

    def fail_pipeline(*args, **kwargs):
        raise AssertionError("pipeline should not run when all documents are empty")

    monkeypatch.setattr(index_module, "parse_input_files", lambda *args, **kwargs: parsed_result(native_documents=documents))
    monkeypatch.setattr(index_module, "AdvancedIngestionPipeline", fail_pipeline)
    monkeypatch.setattr(index_module, "get_save_dir", lambda: "data")
    manager = make_manager()

    result = manager.load_files([{"name": "empty.pdf"}], 512, 64)

    assert result["nodes"] == []
    assert result["quality_report"]["files"][0]["status"] == "bad"
    assert manager.inserted_nodes == []


def test_load_dir_ingests_only_valid_documents(monkeypatch):
    valid_text = "This is valid indexable text. " * 5
    documents = [
        doc("", "mixed.pdf", "1"),
        doc(valid_text, "mixed.pdf", "2"),
    ]
    FakePipeline.seen_documents = None

    monkeypatch.setattr(index_module, "list_input_files", lambda input_dir: ["data/mixed.pdf"])
    monkeypatch.setattr(index_module, "parse_input_files", lambda *args, **kwargs: parsed_result(native_documents=documents))
    monkeypatch.setattr(index_module, "AdvancedIngestionPipeline", FakePipeline)
    manager = make_manager()

    result = manager.load_dir("data", 512, 64)

    assert result["nodes"] == ["node-1"]
    assert len(FakePipeline.seen_documents) == 1
    assert FakePipeline.seen_documents[0].text.startswith("This is valid")
    assert result["quality_report"]["files"][0]["status"] == "warning"
    assert manager.inserted_nodes == [["node-1"]]


def test_load_files_attaches_parse_report_after_api_parse(monkeypatch):
    quality_documents = [doc("", "scan.pdf", "1")]
    api_node = doc("API parsed text that is long enough to index. " * 3, "scan.pdf", "1")
    api_node.metadata["parse_provider"] = "dashscope_parse"
    FakePipeline.seen_documents = None
    FakePipeline.seen_nodes = None
    parse_report = {
        "candidate_pages": 1,
        "repaired_pages": 1,
        "failed_pages": 0,
        "files": [
            {
                "file_name": "scan.pdf",
                "provider": "dashscope_parse",
                "source": "dashscope_parse",
                "status": "success",
                "candidate_pages": 1,
                "repaired_pages": 1,
                "failed_pages": 0,
            }
        ],
    }

    monkeypatch.setattr(
        index_module,
        "parse_input_files",
        lambda *args, **kwargs: parsed_result(api_nodes=[api_node], quality_documents=quality_documents, parse_report=parse_report),
    )
    monkeypatch.setattr(index_module, "AdvancedIngestionPipeline", FakePipeline)
    monkeypatch.setattr(index_module, "get_save_dir", lambda: "data")
    manager = make_manager()

    result = manager.load_files([{"name": "scan.pdf"}], 512, 64)

    assert result["nodes"] == ["node-1"]
    assert FakePipeline.seen_nodes[0].metadata["parse_provider"] == "dashscope_parse"
    assert result["quality_report"]["summary"]["parse_repaired_pages"] == 1
    assert result["quality_report"]["files"][0]["parse_repaired_pages"] == 1


def test_rebuild_without_nodes_keeps_existing_storage(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    storage_dir = tmp_path / "storage"
    storage_dir.mkdir()
    (storage_dir / "docstore.json").write_text("old", encoding="utf-8")

    manager = IndexManager.__new__(IndexManager)
    manager.storage_context = object()
    manager.persist_dir = "storage"
    manager.index = "old-index"
    manager.index_id = "old-id"
    monkeypatch.setattr(
        manager,
        "load_dir",
        lambda input_dir, chunk_size, chunk_overlap: {"nodes": [], "quality_report": build_quality_report([])},
    )

    result = manager._rebuild_dir_with_staging("data", 512, 64)

    assert result["nodes"] == []
    assert (storage_dir / "docstore.json").read_text(encoding="utf-8") == "old"
    assert not (tmp_path / "storage_staging").exists()
    assert manager.index == "old-index"
    assert manager.index_id == "old-id"


def test_commit_staged_storage_backs_up_old_index_and_preserves_config(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    storage_dir = tmp_path / "storage"
    staging_dir = tmp_path / "storage_staging"
    storage_dir.mkdir()
    staging_dir.mkdir()
    (storage_dir / "docstore.json").write_text("old-docstore", encoding="utf-8")
    (storage_dir / "config_store.json").write_text("config", encoding="utf-8")
    (staging_dir / "docstore.json").write_text("new-docstore", encoding="utf-8")
    (staging_dir / "index_store.json").write_text("new-index", encoding="utf-8")

    manager = IndexManager.__new__(IndexManager)
    manager._commit_staged_storage(staging_dir)

    assert (storage_dir / "docstore.json").read_text(encoding="utf-8") == "new-docstore"
    assert (storage_dir / "index_store.json").read_text(encoding="utf-8") == "new-index"
    assert (storage_dir / "config_store.json").read_text(encoding="utf-8") == "config"
    assert list((storage_dir / "backups").glob("*/docstore.json"))
    assert not staging_dir.exists()
