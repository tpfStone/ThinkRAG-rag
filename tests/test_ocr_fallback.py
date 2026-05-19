from types import SimpleNamespace

import server.ocr.fallback as fallback_module
from server.ocr.base import OcrPageResult
from server.ocr.fallback import apply_ocr_fallback


def doc(text, file_path="scan.pdf", page="1", file_type="application/pdf"):
    return SimpleNamespace(
        text=text,
        metadata={
            "file_name": "scan.pdf",
            "file_path": file_path,
            "file_type": file_type,
            "page_label": page,
        },
    )


class FakeOcrEngine:
    def __init__(self, text):
        self.text = text
        self.calls = []

    def recognize_image(self, image_path):
        self.calls.append(image_path)
        return OcrPageResult(
            text=self.text,
            confidence=0.93,
            engine="fake-ocr",
            model="fake-model",
            lang="ch",
        )


def test_ocr_fallback_repairs_empty_pdf_page(monkeypatch, tmp_path):
    pdf_path = tmp_path / "scan.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"image")
    document = doc("", file_path=str(pdf_path))
    ocr_text = "OCR repaired Chinese and English text. " * 4
    engine = FakeOcrEngine(ocr_text)

    monkeypatch.setattr(fallback_module, "render_pdf_page_to_png", lambda *args, **kwargs: str(image_path))

    documents, report = apply_ocr_fallback([document], enabled=True, engine=engine)

    assert documents[0].text.startswith("OCR repaired")
    assert documents[0].metadata["extraction_method"] == "ocr"
    assert documents[0].metadata["ocr_engine"] == "fake-ocr"
    assert report["candidate_pages"] == 1
    assert report["repaired_pages"] == 1
    assert report["failed_pages"] == 0


def test_ocr_fallback_does_not_touch_native_text(monkeypatch, tmp_path):
    pdf_path = tmp_path / "native.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    document = doc("This page already has enough native text for indexing. " * 3, file_path=str(pdf_path))
    engine = FakeOcrEngine("should not be used")

    documents, report = apply_ocr_fallback([document], enabled=True, engine=engine)

    assert documents[0].metadata["extraction_method"] == "native"
    assert engine.calls == []
    assert report["candidate_pages"] == 0


def test_ocr_fallback_marks_missing_pdf_as_failure():
    document = doc("", file_path="missing.pdf")

    documents, report = apply_ocr_fallback([document], enabled=True, engine=FakeOcrEngine("unused"))

    assert documents[0].metadata["extraction_method"] == "ocr_failed"
    assert report["candidate_pages"] == 1
    assert report["failed_pages"] == 1
