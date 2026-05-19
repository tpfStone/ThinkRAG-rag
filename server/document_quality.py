from __future__ import annotations

import os
import re
import unicodedata
from collections import defaultdict
from typing import Any


MIN_PAGE_TEXT_LEN = 50
WEIRD_RATIO_THRESHOLD = 0.05
READABLE_RATIO_THRESHOLD = 0.5
AVG_TEXT_LEN_WARNING = 100

BAD_PAGE_STATUSES = {"empty", "too_short", "garbled", "low_readability"}
INDEXABLE_PAGE_STATUSES = {"ok"}
API_PARSE_PAGE_STATUSES = {"empty", "too_short"}

_HYPHENATED_LINE_BREAK_RE = re.compile(r"([A-Za-z]+)-[ \t]*\n[ \t]*([a-z]+)")
_HORIZONTAL_SPACE_RE = re.compile(r"[ \t\f\v]+")
_EXCESSIVE_NEWLINES_RE = re.compile(r"\n{3,}")
_SUSPICIOUS_CHARS = {
    "\ufffd",  # replacement char
    "\u00e2",  # a common mojibake marker
    "\u00c2",
    "\u00c3",
    "\u00e5",
    "\u00bc",
    "\u20ac",
    "\u00ef",
    "\u00bf",
    "\u00bd",
}
_ASCII_PUNCTUATION = set("""!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~""")
_CJK_PUNCTUATION_RANGES = (
    ("\u3000", "\u303f"),
    ("\uff00", "\uffef"),
)


def clean_text(text: str | None) -> str:
    if text is None:
        return ""

    cleaned = unicodedata.normalize("NFKC", str(text))
    cleaned = cleaned.replace("\x00", "")
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = _HYPHENATED_LINE_BREAK_RE.sub(r"\1\2", cleaned)

    lines = []
    for line in cleaned.split("\n"):
        line = _HORIZONTAL_SPACE_RE.sub(" ", line).strip()
        lines.append(line)

    cleaned = "\n".join(lines)
    cleaned = _EXCESSIVE_NEWLINES_RE.sub("\n\n", cleaned)
    return cleaned.strip()


def assess_page_quality(text: str | None) -> dict[str, Any]:
    cleaned = clean_text(text)
    text_len = len(cleaned)

    if text_len == 0:
        status = "empty"
    elif text_len < MIN_PAGE_TEXT_LEN:
        status = "too_short"
    else:
        weird_ratio = _ratio(_count_weird_chars(cleaned), text_len)
        readable_ratio = _ratio(_count_readable_chars(cleaned), text_len)
        if weird_ratio > WEIRD_RATIO_THRESHOLD:
            status = "garbled"
        elif readable_ratio < READABLE_RATIO_THRESHOLD:
            status = "low_readability"
        else:
            status = "ok"

    weird_ratio = _ratio(_count_weird_chars(cleaned), text_len)
    readable_ratio = _ratio(_count_readable_chars(cleaned), text_len)
    return {
        "status": status,
        "text_len": text_len,
        "weird_ratio": weird_ratio,
        "readable_ratio": readable_ratio,
        "needs_api_parse": status in API_PARSE_PAGE_STATUSES,
    }


def build_quality_report(documents: list[Any]) -> dict[str, Any]:
    page_results = _analyze_documents(documents, clean_documents=False)
    return _build_report_from_page_results(page_results)


def filter_indexable_documents(documents: list[Any]) -> list[Any]:
    indexable_documents, _ = prepare_documents_for_indexing(documents)
    return indexable_documents


def should_api_parse_page(text: str | None, quality: dict[str, Any] | None = None) -> bool:
    quality = quality or assess_page_quality(text)
    return quality.get("status") in API_PARSE_PAGE_STATUSES


def prepare_documents_for_indexing(documents: list[Any]) -> tuple[list[Any], dict[str, Any]]:
    page_results = _analyze_documents(documents, clean_documents=True)
    report = _build_report_from_page_results(page_results)
    indexable_documents = [
        item["document"]
        for item in page_results
        if item["status"] in INDEXABLE_PAGE_STATUSES
    ]
    return indexable_documents, report


def _analyze_documents(documents: list[Any], clean_documents: bool) -> list[dict[str, Any]]:
    page_results = []
    for document in documents or []:
        raw_text = _get_document_text(document)
        cleaned_text = clean_text(raw_text)
        quality = assess_page_quality(cleaned_text)
        if clean_documents:
            _set_document_text(document, cleaned_text)

        metadata = _get_document_metadata(document)
        file_name = _document_file_name(document, metadata)
        page_label = metadata.get("page_label") or metadata.get("page") or "N/A"
        page_results.append(
            {
                "document": document,
                "file_name": file_name,
                "page_label": page_label,
                "status": quality["status"],
                "text_len": quality["text_len"],
                "weird_ratio": quality["weird_ratio"],
                "readable_ratio": quality["readable_ratio"],
                "needs_api_parse": quality["needs_api_parse"],
            }
        )
    return page_results


def _build_report_from_page_results(page_results: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in page_results:
        grouped[item["file_name"]].append(item)

    files = []
    total_indexable = 0
    total_skipped = 0
    for file_name, items in sorted(grouped.items()):
        total_pages = len(items)
        empty_pages = sum(1 for item in items if item["status"] == "empty")
        bad_pages = sum(1 for item in items if item["status"] in BAD_PAGE_STATUSES)
        indexable_pages = sum(1 for item in items if item["status"] in INDEXABLE_PAGE_STATUSES)
        avg_text_len = round(sum(item["text_len"] for item in items) / total_pages) if total_pages else 0
        empty_ratio = _ratio(empty_pages, total_pages)
        bad_ratio = _ratio(bad_pages, total_pages)

        if empty_ratio > 0.5:
            status = "bad"
            reason = "too_many_empty_pages"
        elif bad_ratio > 0.5:
            status = "bad"
            reason = "too_many_low_quality_pages"
        elif avg_text_len < AVG_TEXT_LEN_WARNING:
            status = "warning"
            reason = "low_text_density"
        elif bad_pages > 0:
            status = "warning"
            reason = "some_pages_low_quality"
        else:
            status = "ok"
            reason = "extractable_text"

        needs_api_parse = status == "bad" and (empty_ratio > 0.5 or avg_text_len < MIN_PAGE_TEXT_LEN)
        total_indexable += indexable_pages
        total_skipped += total_pages - indexable_pages
        files.append(
            {
                "file_name": file_name,
                "status": status,
                "reason": reason,
                "total_pages": total_pages,
                "empty_pages": empty_pages,
                "bad_pages": bad_pages,
                "avg_text_len": avg_text_len,
                "needs_api_parse": needs_api_parse,
            }
        )

    summary = {
        "total_files": len(files),
        "ok_files": sum(1 for item in files if item["status"] == "ok"),
        "warning_files": sum(1 for item in files if item["status"] == "warning"),
        "bad_files": sum(1 for item in files if item["status"] == "bad"),
        "indexable_documents": total_indexable,
        "skipped_documents": total_skipped,
    }
    return {"files": files, "summary": summary}


def _get_document_text(document: Any) -> str:
    text = getattr(document, "text", None)
    if text is not None:
        return str(text)
    if hasattr(document, "get_content"):
        try:
            return str(document.get_content())
        except Exception:
            return ""
    return ""


def _set_document_text(document: Any, text: str) -> None:
    try:
        setattr(document, "text", text)
    except Exception:
        pass


def _get_document_metadata(document: Any) -> dict[str, Any]:
    metadata = getattr(document, "metadata", None) or getattr(document, "extra_info", None) or {}
    return metadata if isinstance(metadata, dict) else {}


def _document_file_name(document: Any, metadata: dict[str, Any]) -> str:
    file_name = metadata.get("file_name") or metadata.get("title")
    if file_name:
        return str(file_name)

    file_path = metadata.get("file_path")
    if file_path:
        return os.path.basename(str(file_path))

    document_id = getattr(document, "id_", None) or getattr(document, "doc_id", None)
    return str(document_id or "unknown")


def _count_weird_chars(text: str) -> int:
    return sum(1 for char in text if char in _SUSPICIOUS_CHARS)


def _count_readable_chars(text: str) -> int:
    return sum(1 for char in text if _is_readable_char(char))


def _is_readable_char(char: str) -> bool:
    if char.isspace():
        return True
    if "A" <= char <= "Z" or "a" <= char <= "z" or "0" <= char <= "9":
        return True
    if "\u4e00" <= char <= "\u9fff":
        return True
    if char in _ASCII_PUNCTUATION:
        return True
    return any(start <= char <= end for start, end in _CJK_PUNCTUATION_RANGES)


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0
