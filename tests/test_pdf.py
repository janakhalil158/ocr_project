"""
Tests for Phase 1: PDF loading, type detection, and routing.

Uses PyMuPDF itself to programmatically build small temporary PDFs
(text-only, scanned-simulated via embedded images, and mixed) so the
tests don't depend on external fixture files.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf as fitz  # PyMuPDF (using the non-deprecated import name)
import pytest

from src.pdf.loader import PDFLoadError, get_pdf_metadata, load_pdf
from src.pdf.pdf_type_detector import (
    DocumentType,
    PageType,
    detect_pdf_type,
)
from src.pipeline.router import RoutingError, route
from src.utils.config import PDFTypeDetectionConfig


# --------------------------------------------------------------------------
# Fixtures: build small synthetic PDFs on the fly.
# --------------------------------------------------------------------------


def _make_text_page(doc: fitz.Document, text: str = "This is a sample text page with real content.") -> None:
    page = doc.new_page()
    page.insert_text((72, 72), text)


def _make_blank_scanned_like_page(doc: fitz.Document) -> None:
    # A page with no inserted text simulates a scanned/image-only page
    # (no machine-readable text layer), without needing real image assets.
    doc.new_page()


@pytest.fixture
def text_pdf_path(tmp_path: Path) -> Path:
    doc = fitz.open()
    for i in range(3):
        _make_text_page(doc, f"Page {i + 1}: this document has plenty of extractable text content.")
    path = tmp_path / "text_document.pdf"
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def scanned_pdf_path(tmp_path: Path) -> Path:
    doc = fitz.open()
    for _ in range(3):
        _make_blank_scanned_like_page(doc)
    path = tmp_path / "scanned_document.pdf"
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def mixed_pdf_path(tmp_path: Path) -> Path:
    doc = fitz.open()
    _make_text_page(doc, "Page 1 has real extractable text content, plenty of it here.")
    _make_blank_scanned_like_page(doc)
    _make_text_page(doc, "Page 3 also has real extractable text content, plenty of it here.")
    path = tmp_path / "mixed_document.pdf"
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def invalid_pdf_path(tmp_path: Path) -> Path:
    """A file with a .pdf extension but that is not actually a valid PDF."""
    path = tmp_path / "not_really_a_pdf.pdf"
    path.write_text("This is just plain text, not a PDF file structure.")
    return path


# --------------------------------------------------------------------------
# Loader tests
# --------------------------------------------------------------------------


class TestPDFLoader:
    def test_valid_pdf_loads_successfully(self, text_pdf_path: Path) -> None:
        document = load_pdf(text_pdf_path)
        try:
            assert document.page_count == 3
        finally:
            document.close()

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        missing_path = tmp_path / "does_not_exist.pdf"
        with pytest.raises(PDFLoadError):
            load_pdf(missing_path)

    def test_invalid_pdf_raises(self, invalid_pdf_path: Path) -> None:
        with pytest.raises(PDFLoadError):
            load_pdf(invalid_pdf_path)

    def test_non_pdf_extension_raises(self, tmp_path: Path) -> None:
        txt_path = tmp_path / "document.txt"
        txt_path.write_text("hello")
        with pytest.raises(PDFLoadError):
            load_pdf(txt_path)

    def test_metadata_extraction(self, text_pdf_path: Path) -> None:
        metadata = get_pdf_metadata(text_pdf_path)
        assert metadata.filename == "text_document.pdf"
        assert metadata.num_pages == 3
        assert metadata.file_size_bytes > 0


# --------------------------------------------------------------------------
# Type detector tests
# --------------------------------------------------------------------------


class TestPDFTypeDetector:
    def test_text_pdf_classified_correctly(self, text_pdf_path: Path) -> None:
        result = detect_pdf_type(text_pdf_path)
        assert result.document_type is DocumentType.TEXT_PDF
        assert all(p.type is PageType.TEXT for p in result.pages)

    def test_scanned_pdf_classified_correctly(self, scanned_pdf_path: Path) -> None:
        result = detect_pdf_type(scanned_pdf_path)
        assert result.document_type is DocumentType.SCANNED_PDF
        assert all(p.type is PageType.SCANNED for p in result.pages)

    def test_mixed_pdf_classified_correctly(self, mixed_pdf_path: Path) -> None:
        result = detect_pdf_type(mixed_pdf_path)
        assert result.document_type is DocumentType.MIXED_PDF
        assert result.num_text_pages == 2
        assert result.num_scanned_pages == 1

    def test_page_level_classification_details(self, mixed_pdf_path: Path) -> None:
        result = detect_pdf_type(mixed_pdf_path)
        assert len(result.pages) == 3
        assert result.pages[0].page_number == 1
        assert result.pages[0].type is PageType.TEXT
        assert result.pages[0].character_count > 0

        assert result.pages[1].page_number == 2
        assert result.pages[1].type is PageType.SCANNED
        assert result.pages[1].character_count == 0

    def test_configurable_threshold_changes_classification(self, text_pdf_path: Path) -> None:
        # With an extremely high threshold, even real text pages should
        # be classified as SCANNED, proving the threshold is honored.
        strict_config = PDFTypeDetectionConfig(min_text_chars_per_page=100_000)
        result = detect_pdf_type(text_pdf_path, config=strict_config)
        assert result.document_type is DocumentType.SCANNED_PDF


# --------------------------------------------------------------------------
# Router tests
# --------------------------------------------------------------------------


class TestRouter:
    def test_text_pdf_routes_to_extract_text(self) -> None:
        assert route(DocumentType.TEXT_PDF) == "extract_text"

    def test_scanned_pdf_routes_to_render_pages_for_ocr(self) -> None:
        assert route(DocumentType.SCANNED_PDF) == "render_pages_for_ocr"

    def test_mixed_pdf_routes_to_process_pages_individually(self) -> None:
        assert route(DocumentType.MIXED_PDF) == "process_pages_individually"

    def test_end_to_end_routing_from_detection(self, scanned_pdf_path: Path) -> None:
        result = detect_pdf_type(scanned_pdf_path)
        action = route(result.document_type)
        assert action == "render_pages_for_ocr"

    def test_unknown_type_raises_routing_error(self) -> None:
        with pytest.raises(RoutingError):
            route("NOT_A_REAL_TYPE")  # type: ignore[arg-type]
