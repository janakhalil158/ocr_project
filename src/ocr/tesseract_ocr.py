"""
Phase 4: Tesseract OCR engine.

The only concrete :class:`~src.ocr.base.OCREngine` implementation for
now (PaddleOCR is a deliberately deferred future backend — see
``src/ocr/base.py``). Tesseract itself is never called directly
elsewhere in the codebase; every call is centralized here through
``pytesseract``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pytesseract

from src.ocr.base import OCREngine
from src.ocr.models import OCRError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TesseractOCREngine(OCREngine):
    """
    OCR engine backed by the Tesseract executable via ``pytesseract``.

    Tesseract's own executable is a separate, OS-level install (e.g.
    ``brew install tesseract`` on macOS, ``apt install tesseract-ocr``
    on Linux) — ``pytesseract`` is only a thin Python wrapper around
    it and does not bundle it.
    """

    name = "tesseract"

    def __init__(self, psm: int = 3, oem: int = 3, tesseract_cmd: Optional[str] = None) -> None:
        """
        Args:
            psm: Page Segmentation Mode passed to Tesseract on every call.
            oem: OCR Engine Mode passed to Tesseract on every call.
            tesseract_cmd: Optional explicit path to the Tesseract
                executable (e.g. ``/opt/local/bin/tesseract`` on a
                MacPorts install). When ``None``, ``pytesseract`` falls
                back to whatever is on the system ``PATH``, which is
                the right default for machines where Tesseract is
                already discoverable and avoids baking a
                machine-specific path into the code.
        """
        self.psm = psm
        self.oem = oem
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
            logger.info("Tesseract executable path overridden: %s", tesseract_cmd)

    def recognize_raw(self, image: np.ndarray, language: str) -> Dict[str, Any]:
        tess_config = f"--oem {self.oem} --psm {self.psm}"
        logger.info("Running Tesseract OCR (language=%s, psm=%d, oem=%d)", language, self.psm, self.oem)
        try:
            return pytesseract.image_to_data(
                image,
                lang=language,
                config=tess_config,
                output_type=pytesseract.Output.DICT,
            )
        except pytesseract.TesseractNotFoundError as exc:
            raise OCRError(
                "The Tesseract executable was not found. pytesseract is only a "
                "wrapper — the Tesseract OCR engine itself must be installed "
                "separately on this machine (e.g. `brew install tesseract` on "
                "macOS, `apt install tesseract-ocr` on Linux) and be on the "
                "system PATH, or its path must be set via OCRConfig.tesseract_cmd."
            ) from exc
        except pytesseract.TesseractError as exc:
            message = str(exc)
            if "Failed loading language" in message or "language" in message.lower():
                raise OCRError(
                    f"Tesseract could not load language data for '{language}'. "
                    f"Make sure the corresponding .traineddata file(s) are "
                    f"installed (e.g. `apt install tesseract-ocr-ara` for "
                    f"Arabic on Linux, or the equivalent language pack on "
                    f"macOS)."
                ) from exc
            raise OCRError(f"Tesseract failed to process the image: {exc}") from exc
