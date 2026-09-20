"""
Phase 4: OCR engine factory.

Selects an :class:`~src.ocr.base.OCREngine` implementation by name, so
the rest of the pipeline never hardcodes a specific engine and never
imports an engine-specific module directly:

    engine = get_ocr_engine("tesseract")   # works today
    engine = get_ocr_engine("paddleocr")   # will work once implemented

Switching the whole pipeline's OCR engine is then a one-line config
change (``OCRConfig.engine = "paddleocr"``, or the ``OCR_ENGINE``
environment variable) with no other code touched.
"""

from __future__ import annotations

from typing import Optional

from src.ocr.base import OCREngine
from src.ocr.easyocr_ocr import EasyOCREngine, convert_language
from src.ocr.models import OCREngineNotAvailableError
from src.ocr.tesseract_ocr import TesseractOCREngine
from src.utils.config import OCR_CONFIG, OCRConfig

#: Engine names this factory can actually construct today.
SUPPORTED_ENGINES = ("tesseract", "easyocr")

#: Engine names that are recognized as real, future backends — a request
#: for one of these gets a clear "not available yet" error rather than
#: being lumped in with a genuine typo/unknown-name error.
PLANNED_ENGINES = ("paddleocr",)


def get_ocr_engine(name: Optional[str] = None, config: OCRConfig = OCR_CONFIG) -> OCREngine:
    """
    Build an :class:`~src.ocr.base.OCREngine` by name.

    Args:
        name: Engine identifier, case-insensitive (e.g. ``"tesseract"``,
            ``"easyocr"``). Defaults to ``config.engine`` (itself driven
            by the ``OCR_ENGINE`` environment variable) when omitted.
        config: Settings used to construct the engine (Tesseract's
            PSM/OEM/executable path, EasyOCR's language/GPU settings, etc.).

    Returns:
        A ready-to-use :class:`~src.ocr.base.OCREngine`.

    Raises:
        src.ocr.models.OCREngineNotAvailableError: If ``name`` is a
            planned-but-not-yet-implemented engine (e.g.
            ``"paddleocr"``), an unrecognized name, or a recognized-but-
            uninstalled/misconfigured engine (e.g. EasyOCR not
            installed). This is always a controlled, informative error —
            this module never lets an engine-specific import fail deep
            inside the pipeline.
    """
    engine_name = (name or config.engine).strip().lower()

    if engine_name == "tesseract":
        return TesseractOCREngine(psm=config.psm, oem=config.oem, tesseract_cmd=config.tesseract_cmd)

    if engine_name == "easyocr":
        return EasyOCREngine(
            languages=convert_language(config.language),
            gpu=config.easyocr_gpu,
        )

    if engine_name in PLANNED_ENGINES:
        raise OCREngineNotAvailableError(
            f"OCR engine '{engine_name}' is not currently installed or implemented "
            f"in this project. It is a prepared integration point — see "
            f"src/ocr/base.py's OCREngine interface — intended for a future "
            f"GPU-equipped machine, not something this pipeline can run yet. "
            f"Set OCRConfig.engine (or the OCR_ENGINE environment variable) to "
            f"one of {SUPPORTED_ENGINES!r} to proceed."
        )

    raise OCREngineNotAvailableError(
        f"Unknown OCR engine '{engine_name}'. Supported: {SUPPORTED_ENGINES!r}. "
        f"Planned (not yet available): {PLANNED_ENGINES!r}."
    )
