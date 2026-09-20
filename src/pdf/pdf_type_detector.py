"""
Page-by-page PDF type detection.

Classifies each page of a PDF as ``TEXT`` or ``SCANNED`` based on the
amount of machine-readable text PyMuPDF can extract, then rolls those
per-page results up into an overall document classification of
``TEXT_PDF``, ``SCANNED_PDF``, or ``MIXED_PDF``.

Classification is always done per page first — the whole-document
label is derived from the page results, never from a single
file-wide text check — so that hybrid documents (e.g. a text report
with a scanned signature page) are correctly identified as mixed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List

import pymupdf as fitz  # PyMuPDF (using the non-deprecated import name)

from src.pdf.loader import load_pdf
from src.utils.config import PDF_TYPE_DETECTION_CONFIG, PDFTypeDetectionConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PageType(str, Enum):
    """Classification of an individual PDF page."""

    TEXT = "TEXT"
    SCANNED = "SCANNED"


class DocumentType(str, Enum):
    """Overall classification of a PDF document."""

    TEXT_PDF = "TEXT_PDF"
    SCANNED_PDF = "SCANNED_PDF"
    MIXED_PDF = "MIXED_PDF"


@dataclass(frozen=True)
class PageClassification:
    """Classification result for a single page."""

    page_number: int  # 1-indexed, matches human-facing page numbering
    type: PageType
    character_count: int

    def to_dict(self) -> dict:
        """Return a plain-dict representation, e.g. for CLI/JSON output."""
        return {
            "page_number": self.page_number,
            "type": self.type.value,
            "character_count": self.character_count,
        }


@dataclass(frozen=True)
class DocumentClassification:
    """Aggregate classification result for an entire PDF."""

    document_type: DocumentType
    pages: List[PageClassification] = field(default_factory=list)

    @property
    def num_text_pages(self) -> int:
        return sum(1 for p in self.pages if p.type is PageType.TEXT)

    @property
    def num_scanned_pages(self) -> int:
        return sum(1 for p in self.pages if p.type is PageType.SCANNED)


def classify_page(page: fitz.Page, page_number: int, config: PDFTypeDetectionConfig) -> PageClassification:
    """
    Classify a single PDF page as TEXT or SCANNED.

    Args:
        page: The PyMuPDF page object.
        page_number: 1-indexed page number for reporting purposes.
        config: Thresholds controlling classification sensitivity.

    Returns:
        A :class:`PageClassification` for this page.
    """
    raw_text = page.get_text("text")
    character_count = len(raw_text.strip())

    page_type = (
        PageType.TEXT
        if character_count >= config.min_text_chars_per_page
        else PageType.SCANNED
    )

    return PageClassification(
        page_number=page_number,
        type=page_type,
        character_count=character_count,
    )


def _aggregate_document_type(pages: List[PageClassification]) -> DocumentType:
    """Derive the overall document type from individual page results."""
    if not pages:
        # No pages were classifiable; treat conservatively as scanned so
        # downstream routing sends it through OCR rather than skipping it.
        return DocumentType.SCANNED_PDF

    all_text = all(p.type is PageType.TEXT for p in pages)
    all_scanned = all(p.type is PageType.SCANNED for p in pages)

    if all_text:
        return DocumentType.TEXT_PDF
    if all_scanned:
        return DocumentType.SCANNED_PDF
    return DocumentType.MIXED_PDF


def detect_pdf_type(
    file_path: str | Path,
    config: PDFTypeDetectionConfig = PDF_TYPE_DETECTION_CONFIG,
) -> DocumentClassification:
    """
    Analyze every page of a PDF and classify the document as a whole.

    Args:
        file_path: Path to the PDF file.
        config: Thresholds controlling per-page classification
            sensitivity. Defaults to the shared application config.

    Returns:
        A :class:`DocumentClassification` containing the overall
        document type and the per-page breakdown.

    Raises:
        PDFLoadError: If the PDF cannot be opened (propagated from
            :func:`src.pdf.loader.load_pdf`).
    """
    document = load_pdf(file_path)

    try:
        pages: List[PageClassification] = [
            classify_page(document[i], page_number=i + 1, config=config)
            for i in range(document.page_count)
        ]
    finally:
        document.close()

    document_type = _aggregate_document_type(pages)

    logger.info(
        "Classified PDF as %s (%d text pages, %d scanned pages out of %d)",
        document_type.value,
        sum(1 for p in pages if p.type is PageType.TEXT),
        sum(1 for p in pages if p.type is PageType.SCANNED),
        len(pages),
    )

    return DocumentClassification(document_type=document_type, pages=pages)
