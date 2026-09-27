"""
Phase 4: Optical character recognition (OCR) — orchestration layer.

Phase 3 (:mod:`src.preprocessing.preprocessing`) produces a
:class:`~src.preprocessing.preprocessing.ProcessedPage` — a page image
that has already been resized, denoised, deskewed, contrast-enhanced
and/or thresholded as needed. Phase 4 takes that image and turns it
into actual text:

    ProcessedPage.image
           |
           v
    Phase 4: extract_text()  -- this module
           |
           +-- delegates the actual recognition to an OCREngine
           |   (src/ocr/base.py, src/ocr/unlimited_ocr.py)
           |
           v
    PageOCRResult (text + per-word detail + confidence)
           (src/ocr/models.py)

This module deliberately contains no engine-specific code. It reads
the engine-agnostic structured data an :class:`~src.ocr.base.OCREngine`
returns (see that module's docstring for the exact shape) and turns it
into a readable page string, a filtered word list, and a page-level
confidence — logic that is written once here and applies identically
no matter which concrete engine produced the data.

Design goals (mirroring Phase 2/3):
    * Do not resize, denoise, or otherwise modify the image here —
      that is Phase 3's job. This module only reads pixels; it never
      writes them.
    * Stay engine-independent: this module talks to an ``OCREngine``,
      never to ``pytesseract`` directly.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np

from src.ocr.base import OCREngine
from src.ocr.models import (
    BoundingBox,
    ConfidenceLevel,
    ConfidenceThresholds,
    OCREngineNotAvailableError,
    OCRError,
    OCRResult,
    PageOCRResult,
    WordResult,
    confidence_level,
)
from src.utils.config import OCR_CONFIG, OCRConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Re-exported for backward compatibility: earlier versions of this
# module defined BoundingBox/WordResult/PageOCRResult/OCRError
# directly, so existing imports of `from src.ocr.ocr import ...`
# continue to work unchanged after the models moved to their own
# module.
__all__ = [
    "BoundingBox",
    "WordResult",
    "PageOCRResult",
    "OCRResult",
    "OCRError",
    "OCREngineNotAvailableError",
    "OCRConfig",
    "OCREngine",
    "ConfidenceLevel",
    "extract_text",
    "extract_text_safe",
]


def _validate_image(image: np.ndarray) -> np.ndarray:
    """Validate that an array looks like a usable, OCR-able image."""
    if image is None:
        raise OCRError("Received a None image.")
    if not isinstance(image, np.ndarray):
        raise OCRError(f"Expected a numpy.ndarray, got {type(image)}.")
    if image.size == 0 or image.ndim not in (2, 3):
        raise OCRError(f"Image has an invalid shape: {getattr(image, 'shape', None)}")
    if image.ndim == 3 and image.shape[2] not in (3, 4):
        raise OCRError(f"Unsupported number of channels: {image.shape}")
    if image.dtype != np.uint8:
        raise OCRError(f"Expected a uint8 image, got dtype {image.dtype}.")

    height, width = image.shape[:2]
    if height < 3 or width < 3:
        raise OCRError(f"Image is too small to OCR: {image.shape}")

    return image


def _words_from_raw_data(data: Dict, min_confidence: float) -> List[WordResult]:
    """
    Extract word-level results from an engine's structured OCR output.

    That output has one row per item at one of five hierarchy levels
    (1=page, 2=block, 3=paragraph, 4=line, 5=word — Tesseract's own
    convention, which every engine normalizes its output into). Only
    level-5 rows are actual recognized words with real text and a real
    confidence; every other level is a structural row with empty text
    and confidence -1. Filtering to non-empty text at/above
    ``min_confidence`` keeps only genuine words and, since -1 < 0,
    excludes structural rows even at the default threshold without a
    separate special case.
    """
    words: List[WordResult] = []
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        confidence = float(data["conf"][i])
        if not text or confidence < min_confidence:
            continue
        words.append(
            WordResult(
                text=text,
                confidence=confidence,
                bounding_box=BoundingBox(
                    x=int(data["left"][i]),
                    y=int(data["top"][i]),
                    width=int(data["width"][i]),
                    height=int(data["height"][i]),
                ),
            )
        )
    return words


def _reconstruct_text(data: Dict, min_confidence: float) -> str:
    """
    Rebuild a readable page string from an engine's structured OCR output.

    Words are grouped by (block_num, par_num, line_num) — "these words
    are on the same line" — and joined with a single space within a
    line. Lines are joined with a newline, in the order the engine
    already returns them (its natural reading order), rather than
    simply concatenating every word with spaces and losing line
    structure.
    """
    lines: List[str] = []
    current_line_key = None
    current_line_words: List[str] = []

    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        confidence = float(data["conf"][i])
        if not text or confidence < min_confidence:
            continue

        line_key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        if line_key != current_line_key:
            if current_line_words:
                lines.append(" ".join(current_line_words))
            current_line_words = []
            current_line_key = line_key

        current_line_words.append(text)

    if current_line_words:
        lines.append(" ".join(current_line_words))

    return "\n".join(lines)


def _regions_from_raw_data(data: Dict) -> Optional[List[Dict[str, Any]]]:
    """
    Build per-region metadata from an engine's raw output, when that
    engine reports region-level detail beyond the base Tesseract-shaped
    contract: an optional ``region_type`` parallel list, and an
    optional ``table_html`` one (see
    :meth:`~src.ocr.unlimited_ocr.UnlimitedOCREngine.recognize_raw`).
    Returns ``None`` when the engine didn't report this, so
    :func:`extract_text` can omit the keys entirely rather than write
    an empty/misleading structure -- this keeps the orchestration layer
    working unchanged for any engine that only implements the base
    word-level contract.

    Unlike :func:`_words_from_raw_data` (which filters to confident,
    non-empty text for the reconstructed page text/word list), every
    detected region is kept here regardless of whether its text ended
    up empty (e.g. a ``[Non-Text]`` region) -- the layout/region
    information itself is still meaningful even when there's no text
    to show for it.
    """
    if "region_type" not in data:
        return None

    row_count = len(data["text"])
    table_html_list = data.get("table_html") or [None] * row_count

    regions: List[Dict[str, Any]] = []
    for i in range(row_count):
        entry: Dict[str, Any] = {
            "type": data["region_type"][i],
            "text": data["text"][i],
            "bounding_box": {
                "x": int(data["left"][i]),
                "y": int(data["top"][i]),
                "width": int(data["width"][i]),
                "height": int(data["height"][i]),
            },
        }
        if table_html_list[i]:
            entry["table_html"] = table_html_list[i]
        regions.append(entry)

    return regions


def _default_engine(config: OCRConfig) -> OCREngine:
    """Build the configured OCR engine from ``config`` (by default, Unlimited-OCR)."""
    # Local import: src.ocr.factory imports UnlimitedOCREngine directly,
    # and importing it back at module scope here would only add an
    # unnecessary import-order constraint for no benefit — deferring it
    # keeps this module's own import list minimal.
    from src.ocr.factory import get_ocr_engine

    return get_ocr_engine(config.engine, config)


def extract_text(
    image: np.ndarray,
    page_number: int = 1,
    config: OCRConfig = OCR_CONFIG,
    engine: Optional[OCREngine] = None,
) -> PageOCRResult:
    """
    Run OCR on a single page image and return a structured result.

    Args:
        image: Page image to OCR, as an RGB(A) or grayscale uint8
            numpy array — typically ``ProcessedPage.image`` from
            Phase 3. The image is read only; it is never resized or
            otherwise modified here (that responsibility belongs
            entirely to Phase 3).
        page_number: 1-based page number, carried through to the result.
        config: OCR settings (language, confidence thresholds, and
            engine-specific settings such as Unlimited-OCR's model
            name/device/dtype).
        engine: The :class:`~src.ocr.base.OCREngine` to use. Defaults
            to the configured engine (see ``src/ocr/factory.py``) built
            from ``config`` — pass a different engine (e.g. a test
            double) to swap it out without changing any other code in
            this module.

    Returns:
        A :class:`~src.ocr.models.PageOCRResult` with the reconstructed
        page text, the list of recognized words (each with a
        confidence and bounding box), and the mean confidence across
        those words. A page with no words at/above
        ``config.min_confidence`` (e.g. a blank page) returns an empty
        ``text``, an empty ``words`` list, and a ``mean_confidence`` of
        0.0.

    Raises:
        src.ocr.models.OCRError: If ``image`` is invalid, the engine is
            unavailable, or the engine fails to process the image.
    """
    _validate_image(image)
    active_engine = engine or _default_engine(config)

    start = time.perf_counter()
    data = active_engine.recognize_raw(image, config.language)
    elapsed_ms = (time.perf_counter() - start) * 1000

    words = _words_from_raw_data(data, config.min_confidence)
    text = _reconstruct_text(data, config.min_confidence)
    mean_confidence = sum(w.confidence for w in words) / len(words) if words else 0.0

    thresholds = ConfidenceThresholds(
        very_good=config.confidence_very_good_threshold,
        good=config.confidence_good_threshold,
        moderate=config.confidence_moderate_threshold,
    )

    logger.info(
        "Page %d OCR complete: engine=%s language=%s %d word(s), "
        "mean confidence %.1f, %.1fms",
        page_number,
        active_engine.name,
        config.language,
        len(words),
        mean_confidence,
        elapsed_ms,
    )

    # `confidence_available` tells callers whether `mean_confidence`/
    # `confidence_level` above are a genuine engine measurement or just
    # the required-but-fabricated 0.0 placeholder (see
    # src.ocr.unlimited_ocr's module docstring on confidence). Defaults
    # to True so any engine that doesn't set
    # `engine_reports_confidence` in its raw dict -- i.e. one that
    # genuinely does report confidence -- keeps its existing behavior.
    metadata: Dict[str, Any] = {
        "confidence_available": bool(data.get("engine_reports_confidence", True)),
    }

    regions = _regions_from_raw_data(data)
    if regions is not None:
        metadata["regions"] = regions
        metadata["tables"] = [r for r in regions if "table_html" in r]

    return PageOCRResult(
        page_number=page_number,
        text=text,
        language=config.language,
        engine=active_engine.name,
        words=words,
        mean_confidence=mean_confidence,
        confidence_level=confidence_level(mean_confidence, thresholds),
        processing_time_ms=elapsed_ms,
        metadata=metadata,
    )


def extract_text_safe(
    image: np.ndarray,
    page_number: int = 1,
    config: OCRConfig = OCR_CONFIG,
    engine: Optional[OCREngine] = None,
) -> Optional[PageOCRResult]:
    """
    Like :func:`extract_text`, but never raises.

    Mirrors the ``*_safe`` convention already used elsewhere in this
    pipeline (e.g.
    :func:`~src.quality.image_quality.assess_page_quality_safe`): on
    any :class:`~src.ocr.models.OCRError` the failure is logged and
    ``None`` is returned instead of propagating, so one page's OCR
    failure doesn't have to crash a whole-document batch run.
    """
    try:
        return extract_text(image, page_number=page_number, config=config, engine=engine)
    except OCRError as exc:
        logger.error("OCR failed for page %d: %s", page_number, exc)
        return None
