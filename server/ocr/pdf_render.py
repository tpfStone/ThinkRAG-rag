from __future__ import annotations

import tempfile
from pathlib import Path


class PdfRenderUnavailable(RuntimeError):
    pass


def render_pdf_page_to_png(pdf_path: str | Path, page_index: int, dpi: int = 300) -> str:
    try:
        import fitz
    except Exception as e:
        raise PdfRenderUnavailable(
            "PyMuPDF is required for OCR fallback. Install optional OCR dependencies with "
            "`pip install pymupdf paddleocr`."
        ) from e

    pdf_path = Path(pdf_path)
    if page_index < 0:
        raise IndexError(f"Invalid PDF page index: {page_index}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        image_path = tmp.name

    document = fitz.open(str(pdf_path))
    try:
        if page_index >= len(document):
            raise IndexError(f"PDF page index {page_index} out of range for {pdf_path}")
        page = document.load_page(page_index)
        pixmap = page.get_pixmap(dpi=dpi, alpha=False)
        pixmap.save(image_path)
    finally:
        document.close()

    return image_path
