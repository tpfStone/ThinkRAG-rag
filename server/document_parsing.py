from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document, TextNode

import config
from server.document_quality import build_quality_report
from server.utils_json import sanitize_for_json


SUPPORTED_NATIVE_SUFFIXES = {
    ".csv",
    ".doc",
    ".docx",
    ".htm",
    ".html",
    ".md",
    ".pdf",
    ".ppt",
    ".pptx",
    ".txt",
}


@dataclass
class ParsedFileResult:
    file_path: Path
    file_hash: str
    native_documents: list[Any] = field(default_factory=list)
    api_nodes: list[Any] = field(default_factory=list)
    quality_documents: list[Any] = field(default_factory=list)
    parse_file_report: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedFilesResult:
    native_documents: list[Any]
    api_nodes: list[Any]
    quality_report: dict[str, Any]
    parse_report: dict[str, Any]


class DocumentParseProvider:
    name = "base"

    def parse_file(
        self,
        file_path: Path,
        file_hash: str,
        chunk_size: int,
        chunk_overlap: int,
        extra_metadata: dict[str, Any] | None = None,
    ) -> ParsedFileResult:
        raise NotImplementedError


class NativeParseProvider(DocumentParseProvider):
    name = "native"

    def parse_file(
        self,
        file_path: Path,
        file_hash: str,
        chunk_size: int,
        chunk_overlap: int,
        extra_metadata: dict[str, Any] | None = None,
    ) -> ParsedFileResult:
        documents = SimpleDirectoryReader(input_files=[str(file_path)]).load_data()
        for document in documents:
            metadata = _ensure_metadata(document)
            metadata.update(
                {
                    "file_hash": file_hash,
                    "parse_provider": self.name,
                    "parse_source": self.name,
                    "parse_status": "success",
                }
            )
            if extra_metadata:
                metadata.update(extra_metadata)
        return ParsedFileResult(
            file_path=file_path,
            file_hash=file_hash,
            native_documents=documents,
            quality_documents=documents,
            parse_file_report={
                "file_name": file_path.name,
                "provider": self.name,
                "source": self.name,
                "status": "success",
                "candidate_pages": 0,
                "repaired_pages": 0,
                "failed_pages": 0,
                "cache_hit": False,
            },
        )


class DashScopeParseProvider(DocumentParseProvider):
    name = "dashscope_parse"

    def __init__(
        self,
        cache_dir: str | os.PathLike[str] | None = None,
        api_key: str | None = None,
        workspace_id: str | None = None,
        category_id: str | None = None,
    ):
        self.cache_dir = Path(cache_dir or config.THINKRAG_PARSE_CACHE_DIR)
        self.api_key = api_key if api_key is not None else config.DASHSCOPE_API_KEY
        self.workspace_id = workspace_id if workspace_id is not None else config.DASHSCOPE_WORKSPACE_ID
        self.category_id = category_id if category_id is not None else config.DASHSCOPE_CATEGORY_ID

    def parse_file(
        self,
        file_path: Path,
        file_hash: str,
        chunk_size: int,
        chunk_overlap: int,
        extra_metadata: dict[str, Any] | None = None,
    ) -> ParsedFileResult:
        metadata = {
            "file_name": file_path.name,
            "file_path": str(file_path),
            "file_hash": file_hash,
            "parse_provider": self.name,
            "parse_status": "success",
            "parse_fmt_type": "DASHSCOPE_DOCMIND",
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        if not self.api_key:
            error = RuntimeError("DASHSCOPE_API_KEY is required for DashScopeParse.")
            self._write_cache(file_hash, self._failed_cache_payload(file_path, file_hash, error))
            return self._failed_result(file_path, file_hash, str(error))

        cache = self._read_cache(file_hash)
        if cache and cache.get("status") == "success" and cache.get("nodes"):
            nodes = _nodes_from_cache(cache.get("nodes", []), metadata)
            return ParsedFileResult(
                file_path=file_path,
                file_hash=file_hash,
                api_nodes=nodes,
                parse_file_report=self._success_report(file_path, cache_hit=True),
            )

        try:
            json_result = cache.get("json_result") if cache and cache.get("status") == "success" else None
            if not json_result:
                json_result = self._call_dashscope(file_path)
            documents = self._documents_from_json_result(json_result, metadata)
            nodes = self._nodes_from_documents(documents, chunk_size, chunk_overlap)
            if not nodes:
                raise RuntimeError("DashScopeParse returned no indexable nodes.")
            cache_payload = self._success_cache_payload(file_path, file_hash, json_result, nodes)
            self._write_cache(file_hash, cache_payload)
            return ParsedFileResult(
                file_path=file_path,
                file_hash=file_hash,
                api_nodes=nodes,
                parse_file_report=self._success_report(file_path, cache_hit=False),
            )
        except Exception as e:
            self._write_cache(file_hash, self._failed_cache_payload(file_path, file_hash, e))
            return self._failed_result(file_path, file_hash, str(e))

    def _call_dashscope(self, file_path: Path) -> list[dict[str, Any]]:
        try:
            from llama_index.readers.dashscope import DashScopeParse
        except Exception:
            from llama_index.readers.dashscope.base import DashScopeParse

        kwargs = {"api_key": self.api_key, "show_progress": False}
        if self.workspace_id:
            kwargs["workspace_id"] = self.workspace_id
        if self.category_id:
            kwargs["category_id"] = self.category_id
        reader = DashScopeParse(**kwargs)
        return reader.get_json_result(str(file_path))

    def _documents_from_json_result(self, json_result: list[dict[str, Any]], metadata: dict[str, Any]) -> list[Document]:
        documents = []
        for index, item in enumerate(json_result or []):
            text = item.get("DASHSCOPE_DOCMIND") or item.get("text") or json.dumps(item, ensure_ascii=False)
            document = Document(text=text, metadata=metadata.copy())
            document.id_ = str(item.get("job_id") or f"{metadata['file_hash']}-{index}")
            documents.append(document)
        return documents

    def _nodes_from_documents(self, documents: list[Document], chunk_size: int, chunk_overlap: int) -> list[Any]:
        try:
            from llama_index.node_parser.relational.dashscope import DashScopeJsonNodeParser
        except Exception:
            try:
                from llama_index.node_parser.dashscope import DashScopeJsonNodeParser
            except Exception:
                from llama_index.node_parser.dashscope.base import DashScopeJsonNodeParser

        parser = DashScopeJsonNodeParser(chunk_size=chunk_size, overlap_size=chunk_overlap)
        nodes = []
        for document in documents:
            if hasattr(parser, "get_nodes_from_node"):
                nodes.extend(parser.get_nodes_from_node(document))
            else:
                nodes.extend(parser.get_nodes_from_documents([document]))
        for node in nodes:
            metadata = _ensure_metadata(node)
            metadata.update(
                {
                    "file_hash": documents[0].metadata.get("file_hash") if documents else "",
                    "parse_provider": self.name,
                    "parse_source": self.name,
                    "parse_status": "success",
                }
            )
        return nodes

    def _cache_path(self, file_hash: str) -> Path:
        return self.cache_dir / f"{file_hash}.json"

    def _read_cache(self, file_hash: str) -> dict[str, Any] | None:
        cache_path = self._cache_path(file_hash)
        if not cache_path.exists():
            return None
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _write_cache(self, file_hash: str, payload: dict[str, Any]) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self._cache_path(file_hash)
        cache_path.write_text(json.dumps(sanitize_for_json(payload), ensure_ascii=False, indent=2), encoding="utf-8")

    def _success_cache_payload(
        self,
        file_path: Path,
        file_hash: str,
        json_result: list[dict[str, Any]],
        nodes: list[Any],
    ) -> dict[str, Any]:
        stat = file_path.stat()
        return {
            "provider": self.name,
            "status": "success",
            "file_name": file_path.name,
            "file_hash": file_hash,
            "file_size": stat.st_size,
            "file_mtime": stat.st_mtime,
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "json_result": json_result,
            "nodes": _nodes_to_cache(nodes),
            "error_type": None,
            "error_message": None,
        }

    def _failed_cache_payload(self, file_path: Path, file_hash: str, error: Exception) -> dict[str, Any]:
        stat = file_path.stat()
        return {
            "provider": self.name,
            "status": "failed",
            "file_name": file_path.name,
            "file_hash": file_hash,
            "file_size": stat.st_size,
            "file_mtime": stat.st_mtime,
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "json_result": [],
            "nodes": [],
            "error_type": type(error).__name__,
            "error_message": str(error),
        }

    def _success_report(self, file_path: Path, cache_hit: bool) -> dict[str, Any]:
        return {
            "file_name": file_path.name,
            "provider": self.name,
            "source": "cache" if cache_hit else self.name,
            "status": "success",
            "candidate_pages": 0,
            "repaired_pages": 0,
            "failed_pages": 0,
            "cache_hit": cache_hit,
        }

    def _failed_result(self, file_path: Path, file_hash: str, message: str) -> ParsedFileResult:
        return ParsedFileResult(
            file_path=file_path,
            file_hash=file_hash,
            parse_file_report={
                "file_name": file_path.name,
                "provider": self.name,
                "source": "failed",
                "status": "failed",
                "candidate_pages": 0,
                "repaired_pages": 0,
                "failed_pages": 0,
                "cache_hit": False,
                "error": message,
            },
        )


class DocMindDirectProvider(DocumentParseProvider):
    name = "docmind_direct"

    def parse_file(self, *args: Any, **kwargs: Any) -> ParsedFileResult:
        raise NotImplementedError("DocMind direct API provider is reserved for a future version.")


def list_input_files(input_dir: str | os.PathLike[str]) -> list[Path]:
    root = Path(input_dir)
    if not root.exists():
        return []
    return [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_NATIVE_SUFFIXES
    ]


def parse_input_files(
    input_files: list[str | os.PathLike[str]],
    chunk_size: int,
    chunk_overlap: int,
    parse_provider: str | None = None,
    dashscope_provider: DashScopeParseProvider | None = None,
    native_provider: NativeParseProvider | None = None,
) -> ParsedFilesResult:
    provider_name = (parse_provider or config.THINKRAG_PARSE_PROVIDER or "auto").lower()
    native_provider = native_provider or NativeParseProvider()
    dashscope_provider = dashscope_provider or DashScopeParseProvider()

    native_documents_for_indexing: list[Any] = []
    quality_documents: list[Any] = []
    api_nodes: list[Any] = []
    parse_file_reports: list[dict[str, Any]] = []

    for raw_file in input_files:
        file_path = Path(raw_file)
        file_hash = sha256_file(file_path)
        native_result = native_provider.parse_file(file_path, file_hash, chunk_size, chunk_overlap)
        quality_documents.extend(native_result.quality_documents)

        native_quality_report = build_quality_report(native_result.quality_documents)
        native_file_report = _first_file_report(native_quality_report, file_path.name)
        use_api = _should_use_dashscope(provider_name, file_path, native_file_report)
        candidate_pages = _candidate_pages(native_file_report)

        if use_api:
            api_result = dashscope_provider.parse_file(file_path, file_hash, chunk_size, chunk_overlap)
            api_report = api_result.parse_file_report or {}
            api_report["candidate_pages"] = candidate_pages
            if api_result.api_nodes:
                for node in api_result.api_nodes:
                    _ensure_metadata(node).update(
                        {
                            "file_hash": file_hash,
                            "parse_provider": "dashscope_parse",
                            "parse_source": api_report.get("source", "dashscope_parse"),
                            "parse_status": "success",
                        }
                    )
                api_nodes.extend(api_result.api_nodes)
                api_report["repaired_pages"] = candidate_pages
                api_report["failed_pages"] = 0
            else:
                native_documents_for_indexing.extend(native_result.native_documents)
                api_report["repaired_pages"] = 0
                api_report["failed_pages"] = candidate_pages
            parse_file_reports.append(api_report)
        else:
            native_documents_for_indexing.extend(native_result.native_documents)
            native_report = native_result.parse_file_report
            native_report["candidate_pages"] = candidate_pages
            parse_file_reports.append(native_report)

    quality_report = build_quality_report(quality_documents)
    parse_report = _build_parse_report(parse_file_reports, provider_name)
    quality_report = attach_parse_report(quality_report, parse_report)
    return ParsedFilesResult(
        native_documents=native_documents_for_indexing,
        api_nodes=api_nodes,
        quality_report=quality_report,
        parse_report=parse_report,
    )


def attach_parse_report(quality_report: dict[str, Any], parse_report: dict[str, Any]) -> dict[str, Any]:
    quality_report["parse"] = parse_report
    summary = quality_report.setdefault("summary", {})
    summary["parse_candidate_pages"] = int(parse_report.get("candidate_pages", 0))
    summary["parse_attempted_files"] = int(parse_report.get("attempted_files", 0))
    summary["parse_repaired_pages"] = int(parse_report.get("repaired_pages", 0))
    summary["parse_failed_pages"] = int(parse_report.get("failed_pages", 0))

    parse_by_file = {item.get("file_name"): item for item in parse_report.get("files", [])}
    for file_item in quality_report.get("files", []):
        file_report = parse_by_file.get(file_item.get("file_name"), {})
        file_item["parse_provider"] = file_report.get("provider", "native")
        file_item["parse_source"] = file_report.get("source", "native")
        file_item["parse_status"] = file_report.get("status", "success")
        file_item["parse_candidate_pages"] = int(file_report.get("candidate_pages", 0))
        file_item["parse_repaired_pages"] = int(file_report.get("repaired_pages", 0))
        file_item["parse_failed_pages"] = int(file_report.get("failed_pages", 0))
        file_item["parse_cache_hit"] = bool(file_report.get("cache_hit", False))
        if file_report.get("error"):
            file_item["parse_error"] = file_report["error"]

    return quality_report


def sha256_file(file_path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _should_use_dashscope(provider_name: str, file_path: Path, native_file_report: dict[str, Any] | None) -> bool:
    if provider_name == "native":
        return False
    if provider_name == "dashscope_parse":
        return True
    if provider_name != "auto":
        raise ValueError(f"Unsupported THINKRAG_PARSE_PROVIDER: {provider_name}")
    return file_path.suffix.lower() == ".pdf" and bool((native_file_report or {}).get("needs_ocr", False))


def _first_file_report(quality_report: dict[str, Any], file_name: str) -> dict[str, Any] | None:
    for item in quality_report.get("files", []):
        if item.get("file_name") == file_name:
            return item
    files = quality_report.get("files", [])
    return files[0] if files else None


def _candidate_pages(native_file_report: dict[str, Any] | None) -> int:
    if not native_file_report:
        return 0
    return int(native_file_report.get("bad_pages", 0) or 0)


def _build_parse_report(file_reports: list[dict[str, Any]], provider_name: str) -> dict[str, Any]:
    return {
        "provider": provider_name,
        "candidate_pages": sum(int(item.get("candidate_pages", 0) or 0) for item in file_reports),
        "attempted_files": sum(1 for item in file_reports if item.get("provider") == "dashscope_parse"),
        "repaired_pages": sum(int(item.get("repaired_pages", 0) or 0) for item in file_reports),
        "failed_pages": sum(int(item.get("failed_pages", 0) or 0) for item in file_reports),
        "cache_hits": sum(1 for item in file_reports if item.get("cache_hit")),
        "errors": [
            f"{item.get('file_name')}: {item.get('error')}"
            for item in file_reports
            if item.get("error")
        ],
        "files": file_reports,
    }


def _ensure_metadata(document: Any) -> dict[str, Any]:
    metadata = getattr(document, "metadata", None)
    if not isinstance(metadata, dict):
        metadata = getattr(document, "extra_info", None)
    if not isinstance(metadata, dict):
        metadata = {}
        try:
            setattr(document, "metadata", metadata)
        except Exception:
            pass
    return metadata


def _nodes_to_cache(nodes: list[Any]) -> list[dict[str, Any]]:
    cached_nodes = []
    for node in nodes:
        text = getattr(node, "text", None)
        if text is None and hasattr(node, "get_content"):
            text = node.get_content()
        cached_nodes.append(
            {
                "id": getattr(node, "node_id", None) or getattr(node, "id_", None),
                "text": text or "",
                "metadata": _ensure_metadata(node).copy(),
            }
        )
    return cached_nodes


def _nodes_from_cache(cached_nodes: list[dict[str, Any]], metadata: dict[str, Any]) -> list[TextNode]:
    nodes = []
    for item in cached_nodes:
        node_metadata = metadata.copy()
        node_metadata.update(item.get("metadata") or {})
        kwargs = {"text": item.get("text", ""), "metadata": node_metadata}
        if item.get("id"):
            kwargs["id_"] = item["id"]
        nodes.append(TextNode(**kwargs))
    return nodes


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
