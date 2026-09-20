"""
Data-returning orchestration for the document-processing pipeline.

This module exists so a UI (or any other non-CLI caller) can run the
same pipeline ``src/main.py`` runs, and get back structured data
instead of stdout. It does not implement any processing itself — it
only calls the existing Phase 1-4 functions in the order
``print_report`` already uses (see ``src/main.py``) and collects
their results into dataclasses.

Nothing here duplicates OCR, classification, quality-assessment, or
layout logic; every real computation still happens in
``src.pdf.*``, ``src.quality.*``, ``src.preprocessing.*`` and
``src.ocr.*``. Callers that only want the CLI's plain-text report
should keep using ``src/main.py`` — this module is an additional,
non-breaking entry point alongside it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from src.ocr.base import OCREngine
from src.ocr.factory import get_ocr_engine
from src.ocr.models import OCREngineNotAvailableError, PageOCRResult
from src.ocr.ocr import extract_text_safe
from src.pdf.layout_analyzer import PageLayout, analyze_layout
from src.pdf.loader import PDFLoadError, PDFMetadata, get_pdf_metadata
from src.pdf.page_renderer import PageRenderError, render_page_to_array
from src.pdf.pdf_type_detector import (
    DocumentClassification,
    PageType,
    detect_pdf_type,
)
from src.pdf.text_extractor import (
    extract_text_from_pdf_safe,
    is_text_quality_good,
)
from src.preprocessing.preprocessing import ProcessedPage, preprocess_page
from src.quality.image_quality import PageQualityReport, assess_page_quality_safe
from src.utils.config import OCR_CONFIG, PAGE_RENDER_CONFIG
from src.utils.logger import get_logger

logger = get_logger(__name__)

_QUALITY_ASSESSMENT_DPI = PAGE_RENDER_CONFIG.default_dpi

#: Stage identifiers reported to a ``progress_callback``, in pipeline order.
#: A UI can use this list directly to render "PDF -> Classification ->
#: Text Extraction -> Quality Check -> OCR -> Layout Analysis".
PIPELINE_STAGES: tuple[str, ...] = (
    "load",
    "classification",
    "text_extraction",
    "quality_check",
    "ocr",
    "layout_analysis",
    "done",
)

ProgressCallback = Callable[[str, str], None]


class DocumentPipelineError(Exception):
    """Raised when the document cannot be processed at all (e.g. bad PDF)."""


@dataclass
class PageResult:
    """Everything the UI needs to display about a single processed page."""

    page_number: int
    page_type: PageType
    character_count: int

    # Native text extraction (Phase 1).
    native_text: Optional[str] = None
    native_quality_good: Optional[bool] = None

    # OCR (Phase 4), populated when native text was unavailable/unreliable
    # or the page was scanned to begin with.
    used_ocr: bool = False
    ocr_result: Optional[PageOCRResult] = None
    ocr_error: Optional[str] = None

    # Image quality assessment + preprocessing (Phase 2/3), scanned pages only.
    quality_report: Optional[PageQualityReport] = None
    preprocessing: Optional[ProcessedPage] = None

    # Layout (Phase 1 positional reconstruction), native-text pages only.
    layout: Optional[PageLayout] = None

    # Images for display. rendered_image is the raw page render (scanned
    # pages, or a text page that fell back to OCR); processed_image is
    # what preprocessing produced, if preprocessing ran.
    rendered_image: Optional[np.ndarray] = None
    processed_image: Optional[np.ndarray] = None

    error: Optional[str] = None


@dataclass
class DocumentProcessingResult:
    """Full result of running the pipeline on one PDF."""

    metadata: PDFMetadata
    classification: DocumentClassification
    pages: List[PageResult] = field(default_factory=list)
    ocr_engine_name: str = OCR_CONFIG.engine
    ocr_language: str = OCR_CONFIG.language
    ocr_engine_error: Optional[str] = None
    total_processing_time_ms: float = 0.0


def _noop_progress(stage: str, message: str) -> None:
    return None


def _build_ocr_engine() -> tuple[Optional[OCREngine], Optional[str]]:
    """Mirrors main.py's engine construction: never raises, reports the error."""
    try:
        return get_ocr_engine(OCR_CONFIG.engine, OCR_CONFIG), None
    except OCREngineNotAvailableError as exc:
        logger.error("Could not initialize OCR engine %r: %s", OCR_CONFIG.engine, exc)
        return None, str(exc)


def _ocr_page(
    pdf_path: str,
    page_number: int,
    engine: Optional[OCREngine],
    engine_error: Optional[str],
) -> tuple[Optional[np.ndarray], Optional[PageOCRResult], Optional[str]]:
    """Render a page and OCR it, the same way main.py's fallback path does."""
    try:
        image = render_page_to_array(pdf_path, page_number, dpi=_QUALITY_ASSESSMENT_DPI)
    except (PDFLoadError, PageRenderError) as exc:
        return None, None, f"Could not render page {page_number}: {exc}"

    if engine_error is not None:
        return image, None, engine_error

    ocr_result = extract_text_safe(image, page_number=page_number, engine=engine)
    if ocr_result is None:
        return image, None, "OCR failed for this page (see logs for details)."

    return image, ocr_result, None


def process_document(
    pdf_path: str | Path,
    progress_callback: Optional[ProgressCallback] = None,
) -> DocumentProcessingResult:
    """
    Run the full pipeline on one PDF and return structured results.

    Follows the exact same routing ``src/main.py``'s ``print_report``
    uses: classify every page, run native text extraction + a native
    text-quality check on TEXT pages (falling back to OCR when that
    quality check fails), and run quality assessment ->
    preprocessing -> OCR on SCANNED pages. Layout reconstruction runs
    on TEXT pages whose native text was good enough to use.

    Args:
        pdf_path: Path to the PDF to process.
        progress_callback: Optional ``callback(stage, message)``,
            called as the pipeline moves through ``PIPELINE_STAGES``.
            Never required; a UI can use it to drive a progress
            indicator, or omit it to just get the final result.

    Raises:
        DocumentPipelineError: If the PDF itself cannot be loaded or
            classified (mirrors ``print_report``'s own top-level
            error handling).
    """
    progress = progress_callback or _noop_progress
    pdf_path = str(pdf_path)
    start = time.perf_counter()

    progress("load", "Reading PDF metadata")
    try:
        metadata = get_pdf_metadata(pdf_path)
    except PDFLoadError as exc:
        raise DocumentPipelineError(str(exc)) from exc

    progress("classification", "Classifying pages (text vs. scanned)")
    try:
        classification = detect_pdf_type(pdf_path)
    except PDFLoadError as exc:
        raise DocumentPipelineError(str(exc)) from exc

    pages_by_number = {p.page_number: p for p in classification.pages}
    results: dict[int, PageResult] = {
        p.page_number: PageResult(
            page_number=p.page_number,
            page_type=p.type,
            character_count=p.character_count,
        )
        for p in classification.pages
    }

    text_page_numbers = [p.page_number for p in classification.pages if p.type is PageType.TEXT]
    scanned_page_numbers = [p.page_number for p in classification.pages if p.type is PageType.SCANNED]

    ocr_engine, ocr_engine_error = _build_ocr_engine()

    # --- Native text extraction + quality check (TEXT pages) ---
    if text_page_numbers:
        progress("text_extraction", f"Extracting native text from {len(text_page_numbers)} page(s)")
        extraction = extract_text_from_pdf_safe(pdf_path)

        if extraction is None:
            for page_number in text_page_numbers:
                results[page_number].error = "Native text extraction failed."
        else:
            extracted_by_number = {p.page_number: p for p in extraction.pages}

            for page_number in text_page_numbers:
                page_text = extracted_by_number.get(page_number)
                result = results[page_number]

                if page_text is None:
                    result.error = "No native text extraction result for this page."
                    continue

                result.native_text = page_text.text
                quality_good = is_text_quality_good(page_text.text)
                result.native_quality_good = quality_good

                if quality_good:
                    progress("layout_analysis", f"Reconstructing layout for page {page_number}")
                    result.layout = analyze_layout(page_text.words)
                else:
                    progress("ocr", f"Native text unreliable on page {page_number}; falling back to EasyOCR")
                    image, ocr_result, ocr_error = _ocr_page(
                        pdf_path, page_number, ocr_engine, ocr_engine_error
                    )
                    result.used_ocr = True
                    result.rendered_image = image
                    result.ocr_result = ocr_result
                    result.ocr_error = ocr_error

    # --- Quality assessment + preprocessing + OCR (SCANNED pages) ---
    if scanned_page_numbers:
        progress("quality_check", f"Assessing image quality on {len(scanned_page_numbers)} page(s)")

        for page_number in scanned_page_numbers:
            result = results[page_number]

            try:
                image = render_page_to_array(pdf_path, page_number, dpi=_QUALITY_ASSESSMENT_DPI)
            except (PDFLoadError, PageRenderError) as exc:
                result.error = f"Could not render page {page_number}: {exc}"
                continue

            result.rendered_image = image

            report = assess_page_quality_safe(image, page_number=page_number, dpi=_QUALITY_ASSESSMENT_DPI)
            if report is None:
                result.error = "Quality assessment failed."
                continue

            result.quality_report = report

            try:
                processed_page = preprocess_page(image, report)
                result.preprocessing = processed_page
                result.processed_image = processed_page.image

                progress("ocr", f"Running OCR on page {page_number}")
                result.used_ocr = True

                if ocr_engine_error is not None:
                    result.ocr_error = ocr_engine_error
                else:
                    ocr_result = extract_text_safe(processed_page.image, page_number=page_number, engine=ocr_engine)
                    if ocr_result is None:
                        result.ocr_error = "OCR failed for this page (see logs for details)."
                    else:
                        result.ocr_result = ocr_result

            except Exception as exc:  # mirrors main.py's broad catch around preprocessing
                logger.exception("Preprocessing failed for page %d: %s", page_number, exc)
                result.error = f"Preprocessing failed: {exc}"

    progress("done", "Processing complete")
    elapsed_ms = (time.perf_counter() - start) * 1000

    ordered_pages = [results[p.page_number] for p in classification.pages]

    return DocumentProcessingResult(
        metadata=metadata,
        classification=classification,
        pages=ordered_pages,
        ocr_engine_name=OCR_CONFIG.engine,
        ocr_language=OCR_CONFIG.language,
        ocr_engine_error=ocr_engine_error,
        total_processing_time_ms=elapsed_ms,
    )