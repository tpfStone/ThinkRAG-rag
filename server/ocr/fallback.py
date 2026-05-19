from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from config import DATA_DIR, OCR_DPI, OCR_FALLBACK_ENABLED, OCR_LANG
from server.document_quality import assess_page_quality, clean_text, should_ocr_page
from server.ocr.paddle_engine import PaddleOcrEngine, PaddleOcrUnavailable
from server.ocr.pdf_render import PdfRenderUnavailable, render_pdf_page_to_png


def build_empty_ocr_report(enabled: bool | None = None) -> dict[str, Any]:
    return {
        "enabled": OCR_FALLBACK_ENABLED if enabled is None else bool(enabled),
        "engine": "paddleocr",
        "candidate_pages": 0,
        "attempted_pages": 0,
        "repaired_pages": 0,
        "failed_pages": 0,
        "avg_confidence": None,
        "unavailable": False,
        "errors": [],
        "files": [],
    }


def apply_ocr_fallback(
    documents: list[Any],
    enabled: bool | None = None,
    engine: Any | None = None,
    dpi: int | None = None,
) -> tuple[list[Any], dict[str, Any]]:
    enabled = OCR_FALLBACK_ENABLED if enabled is None else bool(enabled)
    dpi = OCR_DPI if dpi is None else int(dpi)
    report = build_empty_ocr_report(enabled=enabled)
    documents = documents or []

    for document in documents:
        metadata = _ensure_metadata(document)
        if _get_document_text(document).strip():
            metadata.setdefault("extraction_method", "native")

    candidates = [
        document
        for document in documents
        if _is_pdf_document(document) and should_ocr_page(_get_document_text(document))
    ]
    report["candidate_pages"] = len(candidates)
    if not enabled or not candidates:
        report["files"] = _file_report_from_counter({})
        return documents, report

    ocr_engine = engine or PaddleOcrEngine(lang=OCR_LANG)
    file_counters: dict[str, dict[str, int]] = defaultdict(
        lambda: {"candidate_pages": 0, "attempted_pages": 0, "repaired_pages": 0, "failed_pages": 0}
    )
    confidences: list[float] = []

    for document in candidates:
        metadata = _ensure_metadata(document)
        file_counters[_metadata_file_name(metadata)]["candidate_pages"] += 1

    for candidate_index, document in enumerate(candidates):
        metadata = _ensure_metadata(document)
        file_name = _metadata_file_name(metadata)
        pdf_path = _metadata_pdf_path(metadata)
        page_index = _metadata_page_index(metadata)
        if pdf_path is None or page_index is None:
            _mark_ocr_failure(document, "missing_pdf_path_or_page")
            _record_failure(report, file_counters[file_name], f"{file_name}: missing PDF path or page label")
            continue

        image_path = None
        try:
            image_path = render_pdf_page_to_png(pdf_path, page_index, dpi=dpi)
            report["attempted_pages"] += 1
            file_counters[file_name]["attempted_pages"] += 1
            ocr_result = ocr_engine.recognize_image(image_path)
            cleaned_text = clean_text(ocr_result.text)
            quality = assess_page_quality(cleaned_text)
            if quality["status"] == "ok":
                _set_document_text(document, cleaned_text)
                metadata.update(
                    {
                        "extraction_method": "ocr",
                        "ocr_engine": ocr_result.engine or "paddleocr",
                        "ocr_model": ocr_result.model or "PP-OCRv5",
                        "ocr_lang": ocr_result.lang or OCR_LANG,
                        "ocr_confidence": ocr_result.confidence,
                        "ocr_dpi": dpi,
                    }
                )
                report["repaired_pages"] += 1
                file_counters[file_name]["repaired_pages"] += 1
                if ocr_result.confidence is not None:
                    confidences.append(float(ocr_result.confidence))
            else:
                _mark_ocr_failure(document, f"ocr_text_quality_{quality['status']}")
                _record_failure(
                    report,
                    file_counters[file_name],
                    f"{file_name} page {page_index + 1}: OCR text quality is {quality['status']}",
                )
        except (PaddleOcrUnavailable, PdfRenderUnavailable) as e:
            report["unavailable"] = True
            _mark_ocr_failure(document, type(e).__name__)
            _record_failure(report, file_counters[file_name], str(e))
            for remaining_document in candidates[candidate_index + 1 :]:
                remaining_file = _metadata_file_name(_ensure_metadata(remaining_document))
                _mark_ocr_failure(remaining_document, type(e).__name__)
                report["failed_pages"] += 1
                file_counters[remaining_file]["failed_pages"] += 1
            break
        except Exception as e:
            _mark_ocr_failure(document, type(e).__name__)
            _record_failure(report, file_counters[file_name], f"{file_name} page {page_index + 1}: {e}")
        finally:
            if image_path:
                _remove_temp_image(image_path)

    report["avg_confidence"] = round(sum(confidences) / len(confidences), 4) if confidences else None
    report["files"] = _file_report_from_counter(file_counters)
    return documents, report


def _record_failure(report: dict[str, Any], file_counter: dict[str, int], message: str) -> None:
    report["failed_pages"] += 1
    file_counter["failed_pages"] += 1
    if message and message not in report["errors"]:
        report["errors"].append(message)


def _mark_ocr_failure(document: Any, reason: str) -> None:
    metadata = _ensure_metadata(document)
    metadata["extraction_method"] = "ocr_failed"
    metadata["ocr_failure_reason"] = reason


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


def _is_pdf_document(document: Any) -> bool:
    metadata = _ensure_metadata(document)
    file_type = str(metadata.get("file_type", "")).lower()
    file_name = _metadata_file_name(metadata).lower()
    file_path = str(metadata.get("file_path", "")).lower()
    return file_type == "application/pdf" or file_name.endswith(".pdf") or file_path.endswith(".pdf")


def _metadata_file_name(metadata: dict[str, Any]) -> str:
    file_name = metadata.get("file_name") or metadata.get("title")
    if file_name:
        return str(file_name)
    file_path = metadata.get("file_path")
    if file_path:
        return Path(str(file_path)).name
    return "unknown"


def _metadata_pdf_path(metadata: dict[str, Any]) -> Path | None:
    file_path = metadata.get("file_path")
    if file_path:
        path = Path(str(file_path))
        if path.exists():
            return path

    file_name = metadata.get("file_name")
    if file_name:
        path = Path(DATA_DIR) / str(file_name)
        if path.exists():
            return path
    return None


def _metadata_page_index(metadata: dict[str, Any]) -> int | None:
    page_label = metadata.get("page_label") or metadata.get("page")
    if page_label is None:
        return None
    try:
        return max(0, int(str(page_label).strip()) - 1)
    except ValueError:
        return None


def _remove_temp_image(image_path: str) -> None:
    try:
        os.remove(image_path)
    except OSError:
        pass


def _file_report_from_counter(counters: dict[str, dict[str, int]]) -> list[dict[str, Any]]:
    return [
        {"file_name": file_name, **counter}
        for file_name, counter in sorted(counters.items(), key=lambda item: item[0])
    ]
