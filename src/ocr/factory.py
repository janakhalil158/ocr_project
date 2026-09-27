"""
Phase 4: OCR engine factory.

Selects the project's single :class:`~src.ocr.base.OCREngine`
implementation, :class:`~src.ocr.unlimited_ocr.UnlimitedOCREngine`.
Kept as a factory function (rather than importing
``UnlimitedOCREngine`` directly wherever an engine is needed) so the
rest of the pipeline still goes through one seam, and so a config- or
environment-driven engine name that isn't ``"unlimited"`` (a typo, or
a name from a since-removed engine such as ``"tesseract"`` or
``"easyocr"``) surfaces a clear, controlled error instead of being
silently ignored or substituted with something the caller didn't ask
for.
"""

from __future__ import annotations

from typing import Optional

from src.ocr.base import OCREngine
from src.ocr.models import OCREngineNotAvailableError
from src.ocr.unlimited_ocr import UnlimitedOCREngine
from src.utils.config import OCR_CONFIG, OCRConfig

#: The only OCR engine name this project supports.
SUPPORTED_ENGINES = ("unlimited",)


def get_ocr_engine(name: Optional[str] = None, config: OCRConfig = OCR_CONFIG) -> OCREngine:
    """
    Build the project's :class:`~src.ocr.base.OCREngine`.

    Args:
        name: Engine identifier, case-insensitive. Only ``"unlimited"``
            is supported; defaults to ``config.engine`` (itself driven
            by the ``OCR_ENGINE`` environment variable) when omitted.
        config: Settings used to construct the engine (Unlimited-OCR's
            model name, device, dtype, and generation length cap).

    Returns:
        A ready-to-use
        :class:`~src.ocr.unlimited_ocr.UnlimitedOCREngine`.

    Raises:
        src.ocr.models.OCREngineNotAvailableError: If ``name`` (or
            ``config.engine``) is anything other than ``"unlimited"``.
            An unrecognized or legacy name (e.g. ``"tesseract"``,
            ``"easyocr"``, ``"paddleocr"``) is a controlled,
            informative failure — never silently ignored or swapped
            for the default engine.
    """
    engine_name = (name or config.engine).strip().lower()

    if engine_name == "unlimited":
        return UnlimitedOCREngine(
            model_name=config.unlimited_model_name,
            device=config.unlimited_device,
            dtype=config.unlimited_dtype,
            max_new_tokens=config.unlimited_max_new_tokens,
        )

    raise OCREngineNotAvailableError(
        f"Unknown OCR engine '{engine_name}'. This project now supports only "
        f"{SUPPORTED_ENGINES!r} (Baidu Unlimited-OCR). Set OCRConfig.engine "
        f"(or the OCR_ENGINE environment variable) to 'unlimited'. If you're "
        f"looking for 'tesseract', 'easyocr', or 'paddleocr' — those engines "
        f"have been removed; Unlimited-OCR is now the project's only "
        f"supported OCR backend."
    )
