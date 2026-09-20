"""
Phase 4: OCR engine interface.

This is the seam that lets the OCR architecture stay engine-independent:

    OCREngine (this module)
         |
         +-- TesseractOCREngine  (src/ocr/tesseract_ocr.py) -- implemented now
         |
         +-- PaddleOCREngine     (future, e.g. on the GPU-equipped PC)
                                     -- not implemented yet, intentionally

Engines are selected by name through :mod:`src.ocr.factory` (e.g.
``get_ocr_engine("tesseract")``), driven by ``OCRConfig.engine`` /
the ``OCR_ENGINE`` environment variable, rather than being hardcoded
anywhere in the pipeline.

Everything above the engine boundary (:mod:`src.ocr.ocr` — word
filtering, text reconstruction, confidence calculation, error
wrapping) works off the single ``recognize_raw`` return shape defined
here, so it never needs to know or care which concrete engine
produced that data. Adding a new engine later means writing one class
that implements :class:`OCREngine`; it does not require touching
preprocessing, PDF handling, quality assessment, NLP, or pipeline
routing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np


class OCREngine(ABC):
    """
    Abstract interface every OCR backend must implement.

    A concrete engine is only responsible for turning a page image
    into Tesseract-shaped structured word data — it does not
    reconstruct readable text, filter tokens, or compute confidence;
    that orchestration is shared and lives in :mod:`src.ocr.ocr` so it
    is written once and applies identically to every engine.

    The rest of the pipeline should never call ``recognize_raw``
    directly or depend on any engine-specific behavior — it should call
    :meth:`process`, the single standardized entry point every engine
    gets for free from this base class.
    """

    #: Short, stable identifier for this engine (e.g. "tesseract").
    #: Used in logging and in :class:`~src.ocr.models.PageOCRResult`
    #: metadata.
    name: str = "unknown"

    @abstractmethod
    def recognize_raw(self, image: np.ndarray, language: str) -> Dict[str, Any]:
        """
        Run OCR on ``image`` and return structured, per-token output.

        The returned dict must use the same shape as
        ``pytesseract.image_to_data(..., output_type=Output.DICT)``:
        a dict of equal-length parallel lists, one entry per detected
        item, with at least the keys ``text``, ``conf``, ``left``,
        ``top``, ``width``, ``height``, ``block_num``, ``par_num``,
        and ``line_num``. This is not a Tesseract-only requirement —
        it is the common contract every engine normalizes its own
        native output into, so :mod:`src.ocr.ocr` can process the
        result the same way regardless of engine.

        Args:
            image: RGB(A) or grayscale uint8 page image. Engines
                should not resize, denoise, or otherwise modify the
                image — that is Phase 3's responsibility.
            language: Engine-specific language configuration string
                (e.g. Tesseract's ``"ara+eng"``).

        Raises:
            src.ocr.models.OCRError: If the engine is unavailable or
                fails to process the image.
        """
        raise NotImplementedError

    def process(self, image: np.ndarray, language: Optional[str] = None, config=None):
        """
        Standardized entry point: run OCR on one page image and return a
        :class:`~src.ocr.models.PageOCRResult` (the project's ``OCRResult``).

        This is what the rest of the pipeline calls —
        ``ocr_engine.process(image)`` — instead of any engine-specific
        function, and instead of calling :meth:`recognize_raw` directly.
        It delegates to the shared orchestration in :mod:`src.ocr.ocr`
        (word filtering, line reconstruction, confidence calculation,
        timing) so that logic is written once and applies identically
        no matter which engine ``self`` is.

        Args:
            image: Page image to OCR (see :meth:`recognize_raw`).
            language: Optional language override. Defaults to
                ``config.language``.
            config: Optional :class:`~src.utils.config.OCRConfig`.
                Defaults to the project-wide ``OCR_CONFIG``.
        """
        # Local import: src.ocr.ocr imports OCREngine from this module,
        # so importing it back at module scope here would be circular.
        from dataclasses import replace

        from src.ocr.ocr import extract_text
        from src.utils.config import OCR_CONFIG

        cfg = config or OCR_CONFIG
        if language is not None:
            cfg = replace(cfg, language=language)
        return extract_text(image, config=cfg, engine=self)
