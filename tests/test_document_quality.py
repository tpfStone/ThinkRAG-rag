from types import SimpleNamespace

from server.document_quality import (
    assess_page_quality,
    build_quality_report,
    clean_text,
    prepare_documents_for_indexing,
    should_ocr_page,
)


def doc(text, file_name="doc.pdf", page="1"):
    return SimpleNamespace(text=text, metadata={"file_name": file_name, "page_label": page})


def test_clean_text_merges_english_hyphenated_line_breaks():
    assert clean_text("retrie-\nval") == "retrieval"
    assert clean_text("multi-\nlingual retrieval") == "multilingual retrieval"


def test_clean_text_does_not_merge_numeric_ranges():
    assert clean_text("2024-\n2025") == "2024-\n2025"


def test_clean_text_normalizes_spacing_and_blank_lines():
    text = "alpha\t\t beta   gamma\n\n\n\nnext"
    assert clean_text(text) == "alpha beta gamma\n\nnext"


def test_assess_page_quality_detects_empty_and_short_text():
    assert assess_page_quality("")["status"] == "empty"
    assert assess_page_quality("short text")["status"] == "too_short"
    assert should_ocr_page("") is True
    assert should_ocr_page("short text") is True


def test_assess_page_quality_detects_garbled_text():
    text = ("This page has readable text " * 10) + ("â" * 40)
    result = assess_page_quality(text)
    assert result["status"] == "garbled"
    assert result["weird_ratio"] > 0.05


def test_assess_page_quality_detects_low_readability():
    text = ("\u2603" * 80) + "abcde"
    result = assess_page_quality(text)
    assert result["status"] == "low_readability"
    assert result["readable_ratio"] < 0.5


def test_build_quality_report_marks_all_empty_pdf_as_bad_and_needs_ocr():
    documents = [
        doc("", "empty.pdf", "1"),
        doc("", "empty.pdf", "2"),
    ]

    report = build_quality_report(documents)

    item = report["files"][0]
    assert item["file_name"] == "empty.pdf"
    assert item["status"] == "bad"
    assert item["reason"] == "too_many_empty_pages"
    assert item["needs_ocr"] is True
    assert report["summary"]["indexable_documents"] == 0
    assert report["summary"]["skipped_documents"] == 2


def test_prepare_documents_for_indexing_keeps_only_valid_pages_and_warns():
    valid_text = "This is a valid page with enough readable text for indexing. " * 3
    documents = [
        doc(valid_text, "mixed.pdf", "1"),
        doc("", "mixed.pdf", "2"),
    ]

    indexable, report = prepare_documents_for_indexing(documents)

    assert len(indexable) == 1
    assert indexable[0].text.startswith("This is a valid page")
    assert report["files"][0]["status"] == "warning"
    assert report["summary"]["indexable_documents"] == 1
    assert report["summary"]["skipped_documents"] == 1
