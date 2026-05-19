from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from server.ocr.base import OcrPageResult


class PaddleOcrUnavailable(RuntimeError):
    pass


class PaddleOcrEngine:
    def __init__(
        self,
        lang: str = "ch",
        text_detection_model_name: str | None = None,
        text_recognition_model_name: str | None = None,
        device: str | None = None,
    ):
        self.lang = lang
        self.text_detection_model_name = text_detection_model_name
        self.text_recognition_model_name = text_recognition_model_name
        self.device = device
        self._ocr = None

    @property
    def engine_name(self) -> str:
        return "paddleocr"

    @property
    def model_name(self) -> str:
        if self.text_recognition_model_name:
            return self.text_recognition_model_name
        return "PP-OCRv5"

    def recognize_image(self, image_path: str) -> OcrPageResult:
        ocr = self._load()
        try:
            if hasattr(ocr, "predict"):
                raw_result = ocr.predict(str(image_path))
                texts, scores = _extract_predict_result(raw_result)
            else:
                raw_result = ocr.ocr(str(image_path), cls=False)
                texts, scores = _extract_legacy_result(raw_result)
        except Exception as e:
            raise PaddleOcrUnavailable(f"PaddleOCR inference failed: {e}") from e

        lines = [text.strip() for text in texts if str(text).strip()]
        confidence = _mean_score(scores)
        return OcrPageResult(
            text="\n".join(lines),
            confidence=confidence,
            engine=self.engine_name,
            model=self.model_name,
            lang=self.lang,
        )

    def _load(self):
        if self._ocr is not None:
            return self._ocr

        try:
            from paddleocr import PaddleOCR
        except Exception as e:
            raise PaddleOcrUnavailable(
                "PaddleOCR is required for OCR fallback. Install optional OCR dependencies with "
                "`pip install paddleocr pymupdf`."
            ) from e

        kwargs: dict[str, Any] = {
            "lang": self.lang,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
        }
        if self.text_detection_model_name:
            kwargs["text_detection_model_name"] = self.text_detection_model_name
        if self.text_recognition_model_name:
            kwargs["text_recognition_model_name"] = self.text_recognition_model_name
        if self.device:
            kwargs["device"] = self.device

        try:
            self._ocr = PaddleOCR(**kwargs)
        except TypeError:
            legacy_kwargs: dict[str, Any] = {"lang": self.lang, "use_angle_cls": False}
            if self.device:
                legacy_kwargs["use_gpu"] = self.device.lower().startswith("gpu")
            try:
                self._ocr = PaddleOCR(**legacy_kwargs)
            except Exception as e:
                raise PaddleOcrUnavailable(f"PaddleOCR initialization failed: {e}") from e
        except Exception as e:
            raise PaddleOcrUnavailable(f"PaddleOCR initialization failed: {e}") from e
        return self._ocr


def _extract_predict_result(raw_result: Any) -> tuple[list[str], list[float]]:
    texts: list[str] = []
    scores: list[float] = []
    for item in _as_iterable(raw_result):
        data = _result_to_dict(item)
        result_data = data.get("res", data) if isinstance(data, dict) else {}
        raw_texts = result_data.get("rec_texts", [])
        raw_scores = result_data.get("rec_scores", [])
        texts.extend(str(text) for text in _as_iterable(raw_texts))
        scores.extend(_float_values(_as_iterable(raw_scores)))
    return texts, scores


def _extract_legacy_result(raw_result: Any) -> tuple[list[str], list[float]]:
    texts: list[str] = []
    scores: list[float] = []
    for page in _as_iterable(raw_result):
        for line in _as_iterable(page):
            if not isinstance(line, (list, tuple)) or len(line) < 2:
                continue
            value = line[1]
            if isinstance(value, (list, tuple)) and value:
                texts.append(str(value[0]))
                if len(value) > 1:
                    scores.extend(_float_values([value[1]]))
    return texts, scores


def _result_to_dict(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return item

    json_value = getattr(item, "json", None)
    if isinstance(json_value, dict):
        return json_value
    if callable(json_value):
        try:
            value = json_value()
            if isinstance(value, dict):
                return value
        except Exception:
            pass

    to_dict = getattr(item, "to_dict", None)
    if callable(to_dict):
        try:
            value = to_dict()
            if isinstance(value, dict):
                return value
        except Exception:
            pass

    return {}


def _as_iterable(value: Any) -> Iterable:
    if value is None:
        return []
    if isinstance(value, (str, bytes, dict)):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def _float_values(values: Iterable[Any]) -> list[float]:
    result: list[float] = []
    for value in values:
        try:
            result.append(float(value))
        except (TypeError, ValueError):
            continue
    return result


def _mean_score(scores: list[float]) -> float | None:
    if not scores:
        return None
    return sum(scores) / len(scores)
