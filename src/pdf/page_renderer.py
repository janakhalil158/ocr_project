"""
PDF page rendering.

Renders selected pages of a PDF to raster images at a configurable
DPI, for later (Phase 2+) OCR/preprocessing consumption. This module
performs rendering ONLY — no OCR, no image preprocessing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np
import pymupdf as fitz  # PyMuPDF (using the non-deprecated import name)

from src.pdf.loader import load_pdf
from src.utils.config import PAGE_RENDER_CONFIG, PageRenderConfig
from src.utils.file_utils import ensure_directory
from src.utils.logger import get_logger

logger = get_logger(__name__)

# PDF points-per-inch, used to convert a target DPI into a PyMuPDF zoom factor.
_POINTS_PER_INCH = 72


class PageRenderError(Exception):
    """Raised when one or more requested pages cannot be rendered."""


def _dpi_to_matrix(dpi: int) -> fitz.Matrix:
    """Convert a DPI value into the zoom matrix PyMuPDF expects."""
    zoom = dpi / _POINTS_PER_INCH
    return fitz.Matrix(zoom, zoom)


def render_pages(
    file_path: str | Path,
    page_numbers: Iterable[int],
    dpi: int = PAGE_RENDER_CONFIG.default_dpi,
    output_dir: str | Path = PAGE_RENDER_CONFIG.output_dir,
    image_format: str = PAGE_RENDER_CONFIG.image_format,
) -> List[Path]:
    """
    Render specified pages of a PDF to image files.

    Args:
        file_path: Path to the source PDF.
        page_numbers: 1-indexed page numbers to render.
        dpi: Render resolution in dots per inch (higher = larger, sharper
            images, at the cost of speed/memory).
        output_dir: Directory that rendered images are written to. It is
            created if it doesn't already exist.
        image_format: Output image format/extension, e.g. ``"png"``.

    Returns:
        List of file paths of the images that were written, in the same
        order as ``page_numbers``.

    Raises:
        PDFLoadError: If the source PDF cannot be opened.
        PageRenderError: If a requested page number is out of range.
    """
    page_numbers = list(page_numbers)
    output_path = ensure_directory(output_dir)
    matrix = _dpi_to_matrix(dpi)

    document = load_pdf(file_path)
    source_name = Path(file_path).stem
    rendered_paths: List[Path] = []

    try:
        for page_number in page_numbers:
            index = page_number - 1  # convert to 0-indexed
            if index < 0 or index >= document.page_count:
                raise PageRenderError(
                    f"Page {page_number} is out of range for a "
                    f"{document.page_count}-page document."
                )

            page = document[index]
            pixmap = page.get_pixmap(matrix=matrix)

            image_filename = f"{source_name}_page_{page_number}.{image_format}"
            image_path = output_path / image_filename
            pixmap.save(str(image_path))

            logger.info(
                "Rendered page %d of '%s' to '%s' at %d DPI",
                page_number,
                source_name,
                image_path,
                dpi,
            )
            rendered_paths.append(image_path)
    finally:
        document.close()

    return rendered_paths


def render_page_to_array(
    file_path: str | Path,
    page_number: int,
    dpi: int = PAGE_RENDER_CONFIG.default_dpi,
) -> np.ndarray:
    """
    Render a single PDF page directly to an in-memory RGB image array.

    This does NOT write anything to disk. It exists for stages (like
    Phase 2 quality assessment) that need to look at pixel data for
    many pages without generating a permanent image file per page —
    important once the pipeline is scoring millions of pages and disk
    I/O for throwaway scoring renders would be wasteful.

    Args:
        file_path: Path to the source PDF.
        page_number: 1-indexed page number to render.
        dpi: Render resolution in dots per inch.

    Returns:
        A ``numpy.ndarray`` of shape ``(height, width, 3)``, dtype
        ``uint8``, in RGB channel order.

    Raises:
        PDFLoadError: If the source PDF cannot be opened.
        PageRenderError: If the requested page number is out of range.
    """
    matrix = _dpi_to_matrix(dpi)
    document = load_pdf(file_path)

    try:
        index = page_number - 1
        if index < 0 or index >= document.page_count:
            raise PageRenderError(
                f"Page {page_number} is out of range for a "
                f"{document.page_count}-page document."
            )

        page = document[index]
        pixmap = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB)

        image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
            pixmap.height, pixmap.width, pixmap.n
        )

        logger.debug(
            "Rendered page %d to in-memory array (%dx%d) at %d DPI",
            page_number,
            pixmap.width,
            pixmap.height,
            dpi,
        )
        # Copy so the array outlives the pixmap/document buffer.
        return image.copy()
    finally:
        document.close()


def render_all_pages(
    file_path: str | Path,
    dpi: int = PAGE_RENDER_CONFIG.default_dpi,
    output_dir: str | Path = PAGE_RENDER_CONFIG.output_dir,
    image_format: str = PAGE_RENDER_CONFIG.image_format,
) -> List[Path]:
    """
    Render every page of a PDF to image files.

    Convenience wrapper around :func:`render_pages` that determines the
    page count automatically.

    Args:
        file_path: Path to the source PDF.
        dpi: Render resolution in dots per inch.
        output_dir: Directory that rendered images are written to.
        image_format: Output image format/extension.

    Returns:
        List of file paths of the images that were written, in page order.
    """
    document = load_pdf(file_path)
    try:
        page_count = document.page_count
    finally:
        document.close()

    return render_pages(
        file_path,
        page_numbers=range(1, page_count + 1),
        dpi=dpi,
        output_dir=output_dir,
        image_format=image_format,
    )
