"""
Phase 4: OCR data models.

Engine-independent value objects shared by every OCR backend and by
the orchestration layer (:mod:`src.ocr.ocr`). Nothing in this module
knows about Tesseract, PaddleOCR, or any other specific engine — that
is deliberate, so the same result shape is produced no matter which
:class:`~src.ocr.base.OCREngine` implementation actually ran.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class OCRError(Exception):
    """Raised when OCR cannot be performed at all (invalid input, engine failure)."""


class OCREngineNotAvailableError(OCRError):
    """
    Raised by the OCR engine factory when a requested engine name is not
    currently installed/implemented on this machine (e.g. "paddleocr"
    before it has been added). This is a controlled, informative failure
    — never a bare ImportError surfacing from deep inside the pipeline.
    """



class ConfidenceLevel(str, Enum):
    """
    Coarse, human-readable bucket for a page's mean OCR confidence.

    This is only an *OCR* confidence indicator — how sure the engine is
    about the characters it produced — not a measure of whether the
    extracted text is semantically correct. A page can score
    VERY_GOOD and still contain OCR mistakes.
    """

    VERY_GOOD = "VERY_GOOD"   # 90-100
    GOOD = "GOOD"             # 75-90
    MODERATE = "MODERATE"     # 50-75
    POOR = "POOR"             # below 50


@dataclass(frozen=True)
class BoundingBox:
    """Pixel-space location of a recognized word on the page image that was passed in."""

    x: int
    y: int
    width: int
    height: int

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}


@dataclass(frozen=True)
class WordResult:
    """A single word recognized by an OCR engine."""

    text: str
    confidence: float
    bounding_box: BoundingBox

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "confidence": round(self.confidence, 2),
            "bounding_box": self.bounding_box.to_dict(),
        }


@dataclass(frozen=True)
class PageOCRResult:
    """
    Full Phase 4 OCR result for a single page — the project's standardized
    ``OCRResult``. Deliberately flexible: ``mean_confidence``,
    ``processing_time_ms``, and ``words`` are all engine-reported values,
    never fabricated when an engine doesn't provide them (Tesseract
    always does; a future engine that doesn't should report ``None``/an
    empty list rather than a made-up number).

    ``language`` is the language configuration the engine was actually
    run with (e.g. ``"ara+eng"``); ``engine`` is that engine's name
    (e.g. ``"unlimited_ocr"``); ``metadata`` carries any remaining
    engine-level detail (for Unlimited-OCR: detected region types,
    table HTML, whether the reported confidence is genuine) that
    callers may want without cluttering the core fields.
    """

    page_number: int
    text: str
    language: str
    engine: str = "unknown"
    words: List[WordResult] = field(default_factory=list)
    mean_confidence: Optional[float] = 0.0
    confidence_level: ConfidenceLevel = ConfidenceLevel.POOR
    processing_time_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def text_detected(self) -> bool:
        """Whether any text was recognized on this page at all."""
        return bool(self.text.strip())

    def to_dict(self) -> dict:
        return {
            "page_number": self.page_number,
            "text": self.text,
            "language": self.language,
            "engine": self.engine,
            "word_count": len(self.words),
            "mean_confidence": round(self.mean_confidence, 2) if self.mean_confidence is not None else None,
            "confidence_level": self.confidence_level.value,
            "text_detected": self.text_detected,
            "processing_time_ms": self.processing_time_ms,
            "metadata": self.metadata,
            "words": [w.to_dict() for w in self.words],
        }


# Alias matching the "OCRResult" name used in the OCR architecture spec —
# PageOCRResult was named for a single page's result before "engine" and
# "processing_time_ms" made it a closer match to that structure; both
# names refer to the same class.
OCRResult = PageOCRResult


@dataclass(frozen=True)
class ConfidenceThresholds:
    """Cutoffs (0-100 scale) used to bucket a page's mean confidence."""

    very_good: float = 90.0
    good: float = 75.0
    moderate: float = 50.0


def confidence_level(
    mean_confidence: float,
    thresholds: Optional[ConfidenceThresholds] = None,
) -> ConfidenceLevel:
    """Bucket a mean OCR confidence (0-100) into a :class:`ConfidenceLevel`."""
    t = thresholds or ConfidenceThresholds()
    if mean_confidence >= t.very_good:
        return ConfidenceLevel.VERY_GOOD
    if mean_confidence >= t.good:
        return ConfidenceLevel.GOOD
    if mean_confidence >= t.moderate:
        return ConfidenceLevel.MODERATE
    return ConfidenceLevel.POOR
