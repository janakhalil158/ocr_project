"""
Phase 4: Baidu Unlimited-OCR engine.

The single concrete :class:`~src.ocr.base.OCREngine` implementation in
this project (see ``src/ocr/factory.py`` -- Tesseract and EasyOCR have
been removed; Unlimited-OCR is now the only supported backend).

Unlimited-OCR (``baidu/Unlimited-OCR`` on Hugging Face) is a
vision-language OCR model, not a classic per-word OCR engine: given a
page image, it returns a single block of text containing tagged
region detections, e.g.::

    <|det|>header [33, 24, 96, 115]<|/det|><Arabic text>
    <|det|>table [32, 244, 969, 697]<|/det|><table>...</table>

Everything above the :class:`~src.ocr.base.OCREngine` boundary
(:mod:`src.ocr.ocr`) still expects the Tesseract-shaped parallel-list
contract described in ``OCREngine.recognize_raw``'s docstring, so this
module's job is entirely translation: parse the model's raw text into
one row per detected region (mirroring the "one region = one row"
approach :mod:`src.ocr.easyocr_ocr` used to take for region-level,
non-word-level output), and attach everything the Tesseract shape has
no room for -- semantic region type, original table HTML -- as extra,
*optional* parallel arrays that :func:`src.ocr.ocr.extract_text` folds
into ``PageOCRResult.metadata`` when present.

Important, deliberate limitations, stated once here rather than hidden:

* Confidence: Unlimited-OCR does not report a confidence score. Every
  row's ``conf`` is the placeholder ``0.0`` -- never a fabricated
  number -- and the raw dict's ``engine_reports_confidence`` is
  ``False`` so callers can tell a genuine 0% apart from "not measured"
  (see :func:`src.ocr.ocr.extract_text`, which turns this into
  ``PageOCRResult.metadata["confidence_available"]``).
* Bounding boxes: Unlimited-OCR reports coordinates on a normalized
  0-1000 grid, not raw pixels. This adapter rescales them against the
  real rendered-page dimensions before returning pixel-space boxes, so
  downstream bounding-box overlays line up with the actual image. Do
  not assume raw pixel coordinates without this conversion.
* Region granularity: each detection is a whole region (a title, a
  paragraph, a table), not a single word -- the same honest limitation
  EasyOCR's own adapter documented for its region-level output.
"""

from __future__ import annotations

import os
import re
import tempfile
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.ocr.base import OCREngine
from src.ocr.models import OCREngineNotAvailableError, OCRError
from src.utils.logger import get_logger

logger = get_logger(__name__)

#: Hugging Face model id. Overridable via OCRConfig.unlimited_model_name.
DEFAULT_MODEL_NAME = "baidu/Unlimited-OCR"

#: The coordinate grid Unlimited-OCR's <|det|> boxes are reported on
#: (normalized 0-1000, independent of the input image's actual pixel
#: size) -- NOT raw pixel coordinates. See module docstring.
COORDINATE_GRID_SIZE = 1000

#: Prompt used to request the model's structured "free OCR" output
#: (region-tagged text + tables). This is a best-effort default based
#: on this model family's documented usage; if your confirmed-working
#: invocation uses a different prompt/mode, override it via
#: OCRConfig / UnlimitedOCREngine(prompt=...) -- only this constant
#: and _run_model() below need to change.
DEFAULT_PROMPT = "<image>\nFree OCR."

_NON_TEXT_MARKER = "[Non-Text]"

_DET_PATTERN = re.compile(
    r"<\|det\|>\s*(?P<type>[A-Za-z_][A-Za-z0-9_]*)\s*"
    r"\[\s*(?P<x1>-?\d+)\s*,\s*(?P<y1>-?\d+)\s*,\s*(?P<x2>-?\d+)\s*,\s*(?P<y2>-?\d+)\s*\]"
    r"\s*<\|/det\|>(?P<content>.*?)(?=<\|det\|>|\Z)",
    re.DOTALL,
)


# --------------------------------------------------------------------------
# Table HTML -> plain text
# --------------------------------------------------------------------------


class _TableToPlainText(HTMLParser):
    """
    Minimal HTML-table -> readable-plain-text converter.

    Deliberately not a general HTML renderer: Unlimited-OCR's table
    output is a simple ``<table><tr><td>...</td></tr></table>``
    structure, so this only needs to track row/cell boundaries. Uses
    only the standard library, so no new dependency is needed just to
    render a table into readable text for the main OCR output.
    """

    def __init__(self) -> None:
        super().__init__()
        self._rows: List[List[str]] = []
        self._current_row: List[str] = []
        self._current_cell: List[str] = []
        self._in_cell = False

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in ("td", "th"):
            self._in_cell = True
            self._current_cell = []
        elif tag == "tr":
            self._current_row = []

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th"):
            self._current_row.append("".join(self._current_cell).strip())
            self._in_cell = False
        elif tag == "tr":
            if self._current_row:
                self._rows.append(self._current_row)

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._current_cell.append(data)

    def as_text(self) -> str:
        return "\n".join("\t".join(cell for cell in row) for row in self._rows)


def html_table_to_text(html: str) -> str:
    """
    Render a Unlimited-OCR ``<table>...</table>`` block as readable
    plain text (tab-separated cells, newline-separated rows).

    Falls back to the raw HTML, stripped, if parsing fails or yields
    nothing -- a malformed table fragment should never crash OCR, and
    the raw HTML is still preserved separately in metadata regardless.
    """
    parser = _TableToPlainText()
    try:
        parser.feed(html)
    except Exception:
        logger.warning("Could not parse table HTML into plain text; falling back to raw HTML.")
        return html.strip()
    text = parser.as_text()
    return text if text else html.strip()


# --------------------------------------------------------------------------
# <|det|> parsing
# --------------------------------------------------------------------------


def _scale_coordinate(value: int, grid_size: int, pixel_size: int) -> int:
    """Rescale one 0-``grid_size`` model coordinate onto 0-``pixel_size`` pixels."""
    if pixel_size <= 0:
        return 0
    scaled = (value / grid_size) * pixel_size
    return max(0, int(round(scaled)))


def parse_detections(raw_output: str, image_height: int, image_width: int) -> List[Dict[str, Any]]:
    """
    Parse Unlimited-OCR's raw ``<|det|>...<|/det|>`` output into a list
    of per-region dicts, one per detection, in the order the model
    produced them (its own reading order).

    Each dict has:
        type: str
            Region type as reported (header/title/text/table/
            page_number/footer/other/...), passed through verbatim
            rather than mapped onto a fixed enum -- the model's set of
            region types is open-ended and every type it produces
            should be preserved, not silently dropped.
        left/top/width/height: int
            Pixel-space box on the *actual* rendered page image,
            converted from the model's normalized 0-1000 coordinates
            (see :data:`COORDINATE_GRID_SIZE`) using ``image_width``/
            ``image_height``. Never left as raw model coordinates.
        text: str
            Text to surface as this region's normal OCR output: the
            recognized text as-is, the table's plain-text rendering
            for table regions, or ``""`` for a ``[Non-Text]`` region
            (never the literal marker string -- see module docstring).
        table_html: Optional[str]
            The original, unmodified HTML for table regions; ``None``
            otherwise.

    Args:
        raw_output: The model's raw generated text for one page image.
        image_height: Height, in pixels, of the image passed to the
            model -- used to rescale the normalized coordinates.
        image_width: Width, in pixels, of the image passed to the model.
    """
    regions: List[Dict[str, Any]] = []

    for match in _DET_PATTERN.finditer(raw_output or ""):
        region_type = match.group("type").strip().lower()
        x1, y1, x2, y2 = (int(match.group(key)) for key in ("x1", "y1", "x2", "y2"))
        content = match.group("content").strip()

        left = _scale_coordinate(min(x1, x2), COORDINATE_GRID_SIZE, image_width)
        top = _scale_coordinate(min(y1, y2), COORDINATE_GRID_SIZE, image_height)
        right = _scale_coordinate(max(x1, x2), COORDINATE_GRID_SIZE, image_width)
        bottom = _scale_coordinate(max(y1, y2), COORDINATE_GRID_SIZE, image_height)

        table_html: Optional[str] = None
        if content == _NON_TEXT_MARKER:
            # Not meaningful OCR text -- keep the region type/box, drop the marker.
            text = ""
        elif region_type == "table" and "<table" in content.lower():
            table_html = content
            text = html_table_to_text(content)
        else:
            text = content

        regions.append(
            {
                "type": region_type,
                "left": left,
                "top": top,
                "width": max(0, right - left),
                "height": max(0, bottom - top),
                "text": text,
                "table_html": table_html,
            }
        )

    return regions


# --------------------------------------------------------------------------
# Model loading (process-wide cache) + engine
# --------------------------------------------------------------------------


def _import_transformers() -> Any:
    """Import ``transformers``, converting a missing install into a controlled error."""
    try:
        import transformers  # type: ignore
    except ImportError as exc:
        raise OCREngineNotAvailableError(
            "The 'transformers' package (and its dependencies: torch, "
            "torchvision, accelerate, einops, addict, easydict) are "
            "required for OCR_ENGINE=unlimited. Install them with "
            "`pip install -r requirements.txt`."
        ) from exc
    return transformers


# The model is loaded at most once per (model_name, device, dtype)
# combination for the life of the process, no matter how many
# UnlimitedOCREngine instances are constructed. The factory builds a
# fresh instance on every get_ocr_engine() call (e.g. once per
# document in src.pipeline.document_pipeline, or once per Streamlit
# rerun) -- this process-wide cache is what actually satisfies
# "load once, reuse across pages/documents" regardless of how many
# times that happens. A Streamlit UI should still additionally wrap
# engine construction in st.cache_resource as its own idiomatic
# caching layer, but correctness here never depends on the caller
# remembering to do that.
_MODEL_CACHE: Dict[Tuple[str, str, str], Tuple[Any, Any]] = {}


class UnlimitedOCREngine(OCREngine):
    """
    OCR engine backed by ``baidu/Unlimited-OCR`` (Hugging Face), loaded
    via ``transformers``.

    The model itself is large (6+ GB) and expensive to load, so it is
    built lazily on first use and cached process-wide (see
    ``_MODEL_CACHE``) rather than per instance -- constructing an
    ``UnlimitedOCREngine`` is cheap; the first call to
    :meth:`recognize_raw` pays the load cost once, and every
    subsequent call (any page, any document, any engine instance with
    the same model/device/dtype) reuses the already-loaded model.
    """

    name = "unlimited_ocr"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "cuda",
        dtype: str = "bfloat16",
        max_new_tokens: int = 8192,
        prompt: str = DEFAULT_PROMPT,
        transformers_module: Optional[Any] = None,
    ) -> None:
        """
        Args:
            model_name: Hugging Face model id to load.
            device: Torch device string (e.g. "cuda", "cuda:0", "cpu").
                Unlimited-OCR is a large multimodal model; GPU
                execution is the supported/expected path, not CPU.
            dtype: Torch dtype name the model weights are loaded in
                (e.g. "bfloat16", "float16", "float32").
            max_new_tokens: Generation length cap passed to the model
                per page.
            prompt: The instruction prompt sent with each page image.
            transformers_module: Optional pre-imported ``transformers``
                module (or a test double exposing compatible
                ``AutoModel``/``AutoTokenizer``). Defaults to a lazy
                real import. Exists so tests never need the real
                package, torch, or the model weights installed.

        Raises:
            src.ocr.models.OCREngineNotAvailableError: If
                ``transformers``/torch aren't installed, or the model
                fails to load.
        """
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.prompt = prompt
        self._transformers_module = transformers_module

    def _cache_key(self) -> Tuple[str, str, str]:
        return (self.model_name, self.device, self.dtype)

    def _get_model_and_tokenizer(self) -> Tuple[Any, Any]:
        """Return this engine's (model, tokenizer), loading/caching them on first use."""
        key = self._cache_key()
        if key in _MODEL_CACHE:
            return _MODEL_CACHE[key]

        transformers = self._transformers_module or _import_transformers()

        logger.info(
            "Loading Unlimited-OCR model %r (device=%s, dtype=%s) -- this "
            "happens once per process.",
            self.model_name,
            self.device,
            self.dtype,
        )
        try:
            import torch  # type: ignore

            torch_dtype = getattr(torch, self.dtype)
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            model = transformers.AutoModel.from_pretrained(
                self.model_name, trust_remote_code=True, use_safetensors=True
            )
            model = model.eval().to(self.device).to(torch_dtype)
        except Exception as exc:  # model loading can raise many exception types
            raise OCREngineNotAvailableError(
                f"Failed to load Unlimited-OCR model {self.model_name!r} on "
                f"device={self.device!r}: {exc}"
            ) from exc

        _MODEL_CACHE[key] = (model, tokenizer)
        return model, tokenizer

    def _run_model(self, image: np.ndarray) -> str:
        """
        Run Unlimited-OCR on one page image and return its raw text
        output (the ``<|det|>``-tagged string described in this
        module's docstring).

        NOTE: This is the single place that calls into the model. If
        your confirmed-working invocation (method name, argument
        names, or how the output text is retrieved) differs from the
        best-effort call below, only this method needs to change --
        nothing else in this file depends on the exact call shape.
        Writes the page to a temp file rather than passing the array
        directly, matching this model family's documented
        ``image_file=`` usage; the temp file is removed immediately
        after inference (not left open for the model to read while a
        Python handle to it is still held, which is unsafe on Windows).
        """
        from PIL import Image

        model, tokenizer = self._get_model_and_tokenizer()
        pil_image = Image.fromarray(image).convert("RGB")

        fd, tmp_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        try:
            pil_image.save(tmp_path)
            try:
                result = model.infer(
                    tokenizer,
                    prompt=self.prompt,
                    image_file=tmp_path,
                    max_new_tokens=self.max_new_tokens,
                )
            except Exception as exc:
                raise OCRError(f"Unlimited-OCR failed to process the image: {exc}") from exc
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        if isinstance(result, dict):
            return str(result.get("text") or result.get("output") or "")
        return str(result)

    def recognize_raw(self, image: np.ndarray, language: str) -> Dict[str, Any]:
        if image.ndim == 2:
            image_height, image_width = image.shape
        else:
            image_height, image_width = image.shape[:2]

        logger.info(
            "Running Unlimited-OCR (model=%s, device=%s, image=%dx%d)",
            self.model_name,
            self.device,
            image_width,
            image_height,
        )

        raw_output = self._run_model(image)
        regions = parse_detections(raw_output, image_height=image_height, image_width=image_width)

        text: List[str] = []
        conf: List[float] = []
        left: List[int] = []
        top: List[int] = []
        width: List[int] = []
        height: List[int] = []
        block_num: List[int] = []
        par_num: List[int] = []
        line_num: List[int] = []
        region_type: List[str] = []
        table_html: List[Optional[str]] = []

        for index, region in enumerate(regions, start=1):
            text.append(region["text"])
            # Unlimited-OCR does not report a confidence score. 0.0 is a
            # placeholder required by the shared Tesseract-shaped
            # schema, never a real measurement -- see
            # engine_reports_confidence below.
            conf.append(0.0)
            left.append(region["left"])
            top.append(region["top"])
            width.append(region["width"])
            height.append(region["height"])
            # No block/paragraph hierarchy is reported; each detected
            # region is its own block/paragraph/line -- deterministic,
            # not fabricated -- the same convention the project's
            # earlier region-level (EasyOCR) adapter used.
            block_num.append(1)
            par_num.append(1)
            line_num.append(index)
            region_type.append(region["type"])
            table_html.append(region["table_html"])

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
            # Extra, optional keys -- not part of the base Tesseract
            # contract, but recognized by src.ocr.ocr.extract_text()
            # when present, and folded into PageOCRResult.metadata so
            # this engine's semantic region types and table HTML
            # survive the shared orchestration layer instead of being
            # discarded.
            "region_type": region_type,
            "table_html": table_html,
            "engine_reports_confidence": False,
        }
