"""
Command-line entry point for the AI Document Processing pipeline.

Pipeline:

PDF loading
    -> page classification
    -> native text extraction when possible
    -> native text quality check
    -> Unlimited-OCR fallback when native extraction is unreliable
    -> image quality assessment for scanned pages
    -> preprocessing
    -> OCR
    -> layout / table reconstruction
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


from src.ocr.base import OCREngine
from src.ocr.factory import get_ocr_engine
from src.ocr.models import (
    OCREngineNotAvailableError,
)
from src.ocr.ocr import extract_text_safe
from src.pdf.layout_analyzer import (
    PageLayout,
    analyze_layout,
)
from src.pdf.loader import (
    PDFLoadError,
    get_pdf_metadata,
)
from src.pdf.page_renderer import (
    PageRenderError,
    render_page_to_array,
)
from src.pdf.pdf_type_detector import (
    PageType,
    detect_pdf_type,
)
from src.pdf.text_extractor import (
    PDFPageText,
    extract_text_from_pdf_safe,
    is_text_quality_good,
)
from src.preprocessing.preprocessing import preprocess_page
from src.quality.image_quality import (
    PageQualityReport,
    assess_page_quality_safe,
)
from src.utils.config import (
    OCR_CONFIG,
    PAGE_RENDER_CONFIG,
)
from src.utils.logger import get_logger


logger = get_logger(__name__)

_QUALITY_ASSESSMENT_DPI = PAGE_RENDER_CONFIG.default_dpi


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "AI Document Processing pipeline: classify PDF, "
            "extract text, OCR, assess image quality, "
            "and reconstruct table layout."
        ),
    )

    parser.add_argument(
        "pdf_path",
        type=str,
        help="Path to the PDF document to process.",
    )

    return parser


def _print_quality_report(
    report: PageQualityReport,
) -> None:

    dpi_display = (
        report.resolution.dpi
        if report.resolution.dpi is not None
        else "UNKNOWN"
    )

    print(f"Page {report.page_number}")
    print(
        f"Resolution: "
        f"{report.resolution.width} x "
        f"{report.resolution.height}"
    )
    print(f"DPI: {dpi_display}")

    print(
        f"Blur: {report.blur.score:.1f} "
        f"({report.blur.status.value})"
    )

    print(
        f"Skew: {report.skew.angle_degrees:.1f}° "
        f"({report.skew.status.value})"
    )

    print(
        f"Contrast: {report.contrast.score:.1f} "
        f"({report.contrast.status.value})"
    )

    print(
        f"Noise: {report.noise.score:.2f} "
        f"({report.noise.status.value})"
    )

    print()
    print(
        f"Overall quality: "
        f"{report.overall_quality.value}"
    )

    if report.recommendations:

        ops = ", ".join(
            report.recommended_operations
        )

        print(
            f"Recommended preprocessing: {ops}"
        )

        for rec in report.recommendations:
            print(
                f"  - {rec.operation.value}: "
                f"{rec.reason}"
            )

    else:
        print(
            "Recommended preprocessing: NONE"
        )

    print()


def _print_layout_analysis(
    layout: PageLayout,
    max_rows: int = 30,
) -> None:

    print("TABLE / LAYOUT ANALYSIS")
    print("-" * 23)

    print(
        f"Rows detected:       "
        f"{len(layout.rows)}"
    )

    print(
        f"Approx. columns:      "
        f"{len(layout.estimated_columns)}"
    )

    print()

    for row_index, row in enumerate(
        layout.rows[:max_rows],
        start=1,
    ):

        print(
            f"Row {row_index}:"
        )

        print(
            " ".join(
                f"[{word.text}]"
                for word in row.words
            )
        )

        print()

    if len(layout.rows) > max_rows:

        print(
            f"... "
            f"({len(layout.rows) - max_rows} "
            f"more row(s) not shown)"
        )

        print()


def _print_reconstructed_table(
    layout: PageLayout,
) -> None:

    print("RECONSTRUCTED TABLE CELLS")
    print("-------------------------")

    if not layout.cells:
        print("No table cells reconstructed.")
        print()
        return

    for row_index, cells in enumerate(
        layout.cells,
        start=1,
    ):

        formatted_cells = []

        for cell in cells:

            formatted_cells.append(
                f"C{cell.column + 1}: "
                f"{cell.text}"
            )

        print(
            f"Row {row_index}: "
            + " | ".join(formatted_cells)
        )

    print()


def _run_ocr(
    image,
    page_number: int,
    engine: Optional[OCREngine] = None,
    engine_error: Optional[
        OCREngineNotAvailableError
    ] = None,
) -> None:

    print("OCR")
    print("---")

    if engine_error is not None:

        print(
            f"Engine: "
            f"{OCR_CONFIG.engine.upper()}"
        )

        print(
            f"OCR failed for this page: "
            f"{engine_error}"
        )

        print()
        return

    ocr_result = extract_text_safe(
        image,
        page_number=page_number,
        engine=engine,
    )

    if ocr_result is None:

        print(
            f"Engine: "
            f"{OCR_CONFIG.engine.upper()}"
        )

        print(
            "OCR failed for this page "
            "(see logs for details)."
        )

        print()
        return

    print(
        f"Engine: "
        f"{ocr_result.engine.upper()}"
    )

    print(
        f"Language: "
        f"{ocr_result.language}"
    )

    print(
        f"Text detected: "
        f"{'YES' if ocr_result.text_detected else 'NO'}"
    )

    print(
        f"Words detected: "
        f"{len(ocr_result.words)}"
    )

    print(
        f"Average confidence: "
        f"{ocr_result.mean_confidence:.2f}% "
        f"({ocr_result.confidence_level.value})"
    )

    print(
        f"Processing time: "
        f"{ocr_result.processing_time_ms:.1f}ms"
    )

    print()

    print("Extracted text:")
    print("-" * 16)

    print(
        ocr_result.text
        if ocr_result.text
        else "(no text recognized)"
    )

    print("-" * 16)
    print()

    return ocr_result


def _run_ocr_fallback(
    pdf_path: str,
    page_number: int,
    engine: Optional[OCREngine],
    engine_error: Optional[
        OCREngineNotAvailableError
    ],
) -> None:

    print("NATIVE TEXT QUALITY CHECK")
    print("-------------------------")
    print(
        "Native extraction appears unreliable."
    )
    print(
        "Falling back to Unlimited-OCR..."
    )
    print()

    try:

        image = render_page_to_array(
            pdf_path,
            page_number,
            dpi=_QUALITY_ASSESSMENT_DPI,
        )

    except (
        PDFLoadError,
        PageRenderError,
    ) as exc:

        print(
            f"Could not render page "
            f"{page_number}: {exc}"
        )

        print()
        return

    return _run_ocr(
        image,
        page_number,
        engine=engine,
        engine_error=engine_error,
    )


def _run_native_text_extraction(
    pdf_path: str,
    text_page_numbers: list[int],
) -> None:

    separator = "=" * 60

    print(separator)
    print("NATIVE TEXT EXTRACTION")
    print(separator)

    extraction = extract_text_from_pdf_safe(
        pdf_path
    )

    if extraction is None:

        print(
            "Native text extraction failed "
            "(see logs for details)."
        )

        print(
            f"{separator}\n"
        )

        return

    relevant_pages = [
        page
        for page in extraction.pages
        if page.page_number in text_page_numbers
    ]

    total_characters = sum(
        page.character_count
        for page in relevant_pages
    )

    print(
        f"Pages extracted:      "
        f"{len(relevant_pages)}"
    )

    print(
        f"Total characters:     "
        f"{total_characters}"
    )

    print(separator)
    print()

    # Build the OCR engine once.
    ocr_engine: Optional[OCREngine] = None
    ocr_engine_error: Optional[
        OCREngineNotAvailableError
    ] = None

    try:

        ocr_engine = get_ocr_engine(
            OCR_CONFIG.engine,
            OCR_CONFIG,
        )

    except OCREngineNotAvailableError as exc:

        ocr_engine_error = exc

        logger.error(
            "Could not initialize OCR engine %r: %s",
            OCR_CONFIG.engine,
            exc,
        )

    for page in relevant_pages:

        print(
            f"PAGE {page.page_number}"
        )

        print("-" * 60)

        print(
            page.text
            if page.text.strip()
            else "(no text extracted)"
        )

        print("-" * 60)
        print()

        quality_good = is_text_quality_good(
            page.text
        )

        print(
            "Native text quality: "
            f"{'GOOD' if quality_good else 'BAD'}"
        )

        print()

        if quality_good:

            print(
                "Using native PDF text."
            )

            print()

            print(
                "Layout information:"
            )

            print(
                f"Words detected: "
                f"{len(page.words)}"
            )

            print()

            layout = analyze_layout(
                page.words
            )

            _print_layout_analysis(
                layout
            )

            _print_reconstructed_table(
                layout
            )

        else:

            _run_ocr_fallback(
                pdf_path,
                page.page_number,
                engine=ocr_engine,
                engine_error=ocr_engine_error,
            )

        print("---")
        print()

    print(
        f"{separator}\n"
    )


def _run_quality_assessment(
    pdf_path: str,
    scanned_page_numbers: list[int],
) -> None:

    separator = "=" * 60

    print(separator)
    print("IMAGE QUALITY ASSESSMENT")
    print("========================")
    print()

    ocr_engine: Optional[OCREngine] = None
    ocr_engine_error: Optional[
        OCREngineNotAvailableError
    ] = None

    try:

        ocr_engine = get_ocr_engine(
            OCR_CONFIG.engine,
            OCR_CONFIG,
        )

    except OCREngineNotAvailableError as exc:

        ocr_engine_error = exc

        logger.error(
            "Could not initialize OCR engine %r: %s",
            OCR_CONFIG.engine,
            exc,
        )

    for page_number in scanned_page_numbers:

        try:

            image = render_page_to_array(
                pdf_path,
                page_number,
                dpi=_QUALITY_ASSESSMENT_DPI,
            )

        except (
            PDFLoadError,
            PageRenderError,
        ) as exc:

            logger.error(
                "Could not render page %d: %s",
                page_number,
                exc,
            )

            print(
                f"Page {page_number}"
            )

            print(
                f"  Skipped: "
                f"could not render page "
                f"({exc})"
            )

            print()

            continue

        report = assess_page_quality_safe(
            image,
            page_number=page_number,
            dpi=_QUALITY_ASSESSMENT_DPI,
        )

        if report is None:

            print(
                f"Page {page_number}"
            )

            print(
                "  Skipped: quality "
                "assessment failed."
            )

            print()

            continue

        _print_quality_report(
            report
        )

        try:

            processed_page = preprocess_page(
                image,
                report,
            )

            print("PREPROCESSING")
            print("=============")

            print(
                f"Page {page_number} "
                f"preprocessing complete."
            )

            print(
                "Recommended operations "
                f"executed: "
                f"{', '.join(report.recommended_operations)}"
            )

            print(
                f"Original image shape: "
                f"{image.shape}"
            )

            print(
                f"Processed image shape: "
                f"{processed_page.image.shape}"
            )

            print()

            _run_ocr(
                processed_page.image,
                page_number,
                engine=ocr_engine,
                engine_error=ocr_engine_error,
            )

        except Exception as exc:

            logger.exception(
                "Preprocessing failed "
                "for page %d: %s",
                page_number,
                exc,
            )

            print(
                f"Page {page_number} "
                f"preprocessing failed: "
                f"{exc}"
            )

            print()

        print("---")
        print()

    print(
        f"{separator}\n"
    )


def print_report(
    pdf_path: str,
) -> int:

    try:

        metadata = get_pdf_metadata(
            pdf_path
        )

        classification = detect_pdf_type(
            pdf_path
        )

    except PDFLoadError as exc:

        print(
            f"\nERROR: {exc}\n",
            file=sys.stderr,
        )

        return 1

    separator = "=" * 60

    print(
        f"\n{separator}"
    )

    print(
        "PDF DOCUMENT PROCESSING "
        "- PHASE 1 REPORT"
    )

    print(separator)

    print(
        f"Filename:            "
        f"{metadata.filename}"
    )

    print(
        f"File size:           "
        f"{metadata.file_size_bytes:,} bytes"
    )

    print(
        f"Number of pages:     "
        f"{metadata.num_pages}"
    )

    print(
        f"Overall classification: "
        f"{classification.document_type.value}"
    )

    print(separator)

    print(
        "PAGE-BY-PAGE CLASSIFICATION"
    )

    print(
        f"{'Page':>6}  "
        f"{'Type':<10}  "
        f"{'Characters':>10}"
    )

    print("-" * 60)

    for page in classification.pages:

        print(
            f"{page.page_number:>6}  "
            f"{page.type.value:<10}  "
            f"{page.character_count:>10}"
        )

    print(separator)

    text_page_numbers = [
        page.page_number
        for page in classification.pages
        if page.type is PageType.TEXT
    ]

    scanned_page_numbers = [
        page.page_number
        for page in classification.pages
        if page.type is PageType.SCANNED
    ]

    if text_page_numbers:

        _run_native_text_extraction(
            pdf_path,
            text_page_numbers,
        )

    if scanned_page_numbers:

        _run_quality_assessment(
            pdf_path,
            scanned_page_numbers,
        )

    return 0


def main() -> None:

    parser = build_arg_parser()

    args = parser.parse_args()

    exit_code = print_report(
        args.pdf_path
    )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()

