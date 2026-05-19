from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OcrPageResult:
    text: str
    confidence: float | None = None
    engine: str = ""
    model: str = ""
    lang: str = ""
