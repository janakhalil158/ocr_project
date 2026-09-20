"""
Phase 4: EasyOCR engine.

A second concrete :class:`~src.ocr.base.OCREngine` implementation,
alongside :class:`~src.ocr.tesseract_ocr.TesseractOCREngine`. Nothing
outside this module knows or needs to know that EasyOCR's native output
shape (``[bounding_box, text, confidence]`` per detected text region) is
different from Tesseract's — this module's only job is to normalize
EasyOCR's results into the same Tesseract-shaped raw dict every engine
produces (see :class:`~src.ocr.base.OCREngine.recognize_raw`'s
docstring), so :mod:`src.ocr.ocr`'s shared orchestration (word
filtering, line reconstruction, confidence averaging) applies to EasyOCR
unchanged.

Honest limitation, stated once here rather than hidden: EasyOCR detects
text region-by-region (often a whole line or phrase per detection), not
word-by-word the way Tesseract does. Rather than inventing a fake
per-word split with fabricated coordinates, each EasyOCR detection is
reported as a single row/"word" in the normalized dict, carrying its own
real bounding box. This means ``WordResult.text`` can be a multi-word
phrase for EasyOCR results, not a single token — an accurate reflection
of what EasyOCR actually detected, not a fabricated one.

EasyOCR itself is imported lazily (inside :func:`_import_easyocr` /
:meth:`EasyOCREngine.__init__`), not at module import time, so importing
this module never requires EasyOCR (or its model downloads) to be
installed, and a missing/broken install surfaces as a controlled
:class:`~src.ocr.models.OCREngineNotAvailableError` rather than a raw
``ImportError`` deep in the pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.ocr.base import OCREngine
from src.ocr.models import OCREngineNotAvailableError, OCRError
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Tesseract-style '+'-joined language tokens -> EasyOCR's own two-letter
# codes. Only the project's current bilingual pair is mapped explicitly;
# see convert_language()'s docstring for how anything else is handled.
_TESSERACT_TO_EASYOCR_LANGUAGE = {
    "eng": "en",
    "ara": "ar",
}


def convert_language(tesseract_language: Optional[str]) -> List[str]:
    """
    Convert a Tesseract-style language string (e.g. ``"ara+eng"``) into
    the list of language codes EasyOCR's ``Reader`` expects.

    Only ``"ara"`` -> ``"ar"`` and ``"eng"`` -> ``"en"`` are mapped
    explicitly, matching the project's current Arabic + English use
    case. Any other token is passed through lowercased rather than
    dropped or raising: EasyOCR already uses two-letter codes matching
    Tesseract's for several languages (e.g. ``"fr"``, ``"de"``), and this
    way a language the caller explicitly asked for is never silently
    discarded just because it isn't in the small explicit map.

    Args:
        tesseract_language: A ``"+"``-joined language string, or ``None``/
            empty for the default.

    Returns:
        A list of EasyOCR language codes, e.g. ``["ar", "en"]``. Falls
        back to :attr:`EasyOCREngine.DEFAULT_LANGUAGES` if
        ``tesseract_language`` is empty or contains no usable tokens.
    """
    if not tesseract_language:
        return list(EasyOCREngine.DEFAULT_LANGUAGES)

    codes: List[str] = []
    for token in tesseract_language.split("+"):
        token = token.strip().lower()
        if not token:
            continue
        mapped = _TESSERACT_TO_EASYOCR_LANGUAGE.get(token, token)
        if mapped not in codes:
            codes.append(mapped)

    return codes or list(EasyOCREngine.DEFAULT_LANGUAGES)


def _import_easyocr() -> Any:
    """
    Import the ``easyocr`` package, converting a missing/broken install
    into a controlled :class:`OCREngineNotAvailableError` instead of
    letting an ``ImportError`` surface from deep inside the pipeline.
    """
    try:
        import easyocr  # type: ignore
    except ImportError as exc:
        raise OCREngineNotAvailableError(
            "EasyOCR is not installed. Install it with `pip install easyocr` "
            "to use OCR_ENGINE=easyocr (or select a different engine)."
        ) from exc
    return easyocr


def _bbox_to_ltwh(points: Sequence[Sequence[float]]) -> tuple[float, float, float, float]:
    """
    Convert EasyOCR's four-corner polygon ``[[x0,y0],[x1,y1],[x2,y2],[x3,y3]]``
    into an axis-aligned ``(left, top, width, height)`` box, matching the
    shape Tesseract's ``image_to_data`` already reports.
    """
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    left = min(xs)
    top = min(ys)
    width = max(xs) - left
    height = max(ys) - top
    return left, top, width, height


class EasyOCREngine(OCREngine):
    """
    OCR engine backed by `EasyOCR <https://github.com/JaidedAI/EasyOCR>`_.

    Unlike Tesseract (a per-call CLI subprocess), EasyOCR's ``Reader``
    loads detection/recognition models the first time it's used, which
    is comparatively expensive. This engine is therefore meant to be
    constructed once (per process/worker) and reused across pages via
    the ``engine=`` parameter :func:`~src.ocr.ocr.extract_text` already
    accepts — not rebuilt on every call. The ``Reader`` itself is
    created lazily on the first :meth:`recognize_raw` call (see
    :meth:`_get_reader`), not at construction time, and is then cached
    on the instance for every subsequent page. See
    :meth:`recognize_raw`'s handling of a mismatched ``language``
    argument for what happens if a caller passes a different language
    than the reader was built for.
    """

    name = "easyocr"

    #: Default EasyOCR language codes, matching the project's current
    #: Arabic + English use case (Tesseract's ``"ara+eng"`` default).
    DEFAULT_LANGUAGES: tuple[str, ...] = ("ar", "en")

    def __init__(
        self,
        languages: Optional[Sequence[str]] = None,
        gpu: bool = False,
        easyocr_module: Optional[Any] = None,
    ) -> None:
        """
        Args:
            languages: EasyOCR language codes (e.g. ``["ar", "en"]``).
                Defaults to :attr:`DEFAULT_LANGUAGES`. Use
                :func:`convert_language` to derive this from the
                project's existing Tesseract-style ``OCRConfig.language``.
            gpu: Whether EasyOCR should use GPU acceleration. Defaults to
                ``False`` (CPU-only) so the pipeline never silently
                requires a GPU to run.
            easyocr_module: Optional pre-imported ``easyocr`` module (or
                a test double exposing a compatible ``Reader`` class).
                Defaults to a lazy real import via :func:`_import_easyocr`.
                Exists so tests can inject a fake module and never need
                the real package installed or its models downloaded.

        Raises:
            src.ocr.models.OCREngineNotAvailableError: If EasyOCR isn't
                installed, or its ``Reader`` fails to initialize.
        """
        self.languages: List[str] = list(languages) if languages else list(self.DEFAULT_LANGUAGES)
        self.gpu = gpu

        # The `easyocr` package import itself was already lazy (deferred
        # to _import_easyocr() rather than a module-level import). The
        # Reader instance -- the expensive part, since it loads
        # detection/recognition models -- is now lazy too: it is NOT
        # built here at construction time, only on the first
        # recognize_raw() call (see _get_reader()). This means
        # constructing an EasyOCREngine is cheap, and a single instance
        # reused across many pages (e.g. by src.main, which now builds
        # one engine per run instead of one per page) pays the reader's
        # model-loading cost exactly once, on whichever page is OCR'd
        # first -- not once per page.
        self._easyocr_module = easyocr_module
        self._reader: Optional[Any] = None

    def _get_reader(self) -> Any:
        """
        Return this engine's EasyOCR ``Reader``, creating it on first use.

        ``self._reader`` is created at most once per engine instance: if
        it already exists, it is returned as-is (no rebuild, no reload
        of the underlying models). If it doesn't exist yet, it is built
        now, from ``self.languages``/``self.gpu`` fixed at construction
        time, and cached on ``self._reader`` for every later call.
        """
        if self._reader is not None:
            return self._reader

        module = (
            self._easyocr_module
            if self._easyocr_module is not None
            else _import_easyocr()
        )

        logger.info(
            "Initializing EasyOCR reader (languages=%s, gpu=%s)",
            self.languages,
            self.gpu,
        )
        try:
            self._reader = module.Reader(self.languages, gpu=self.gpu)
        except Exception as exc:  # EasyOCR/PyTorch can raise varied types here
            raise OCREngineNotAvailableError(
                f"Failed to initialize EasyOCR (languages={self.languages}, "
                f"gpu={self.gpu}): {exc}"
            ) from exc

        return self._reader

    def recognize_raw(self, image: np.ndarray, language: str) -> Dict[str, Any]:
        if language and convert_language(language) != self.languages:
            logger.warning(
                "EasyOCREngine was initialized for languages=%s, but "
                "recognize_raw() was called with language=%r "
                "(-> %s). EasyOCR's language set is fixed at Reader "
                "construction time, not per call, so it is NOT being "
                "rebuilt for this page -- rebuilding it per page would "
                "reload its models on every call. Reconfigure the "
                "engine's languages at construction time if this call's "
                "language should actually be used.",
                self.languages,
                language,
                convert_language(language),
            )

        reader = self._get_reader()

        try:
            raw_results = reader.readtext(image, detail=1)
        except Exception as exc:  # EasyOCR/PyTorch can raise varied types here
            raise OCRError(f"EasyOCR failed to process the image: {exc}") from exc

        text: List[str] = []
        conf: List[float] = []
        left: List[int] = []
        top: List[int] = []
        width: List[int] = []
        height: List[int] = []
        block_num: List[int] = []
        par_num: List[int] = []
        line_num: List[int] = []

        for index, (points, recognized_text, confidence) in enumerate(raw_results, start=1):
            l, t, w, h = _bbox_to_ltwh(points)

            text.append(recognized_text)
            conf.append(float(confidence) * 100.0)  # EasyOCR: 0-1 -> project: 0-100
            left.append(int(round(l)))
            top.append(int(round(t)))
            width.append(int(round(w)))
            height.append(int(round(h)))
            # EasyOCR doesn't report Tesseract's block/paragraph/line
            # hierarchy. Every detection is treated as its own block/
            # paragraph/line (deterministic, not fabricated content) so
            # each stays a distinct row for src.ocr.ocr's line
            # reconstruction rather than being merged with unrelated
            # detections.
            block_num.append(1)
            par_num.append(1)
            line_num.append(index)

        return {
            "text": text,
            "conf": conf,
            "left": left,
            "top": top,
            "width": width,
            "height": height,
            "block_num": block_num,
            "par_num": par_num,
            "line_num": line_num,
        }