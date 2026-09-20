"""
Tests for native PDF text extraction (src/pdf/text_extractor.py).

Builds small temporary PDFs on the fly with PyMuPDF, matching the
existing approach in tests/test_pdf.py, rather than depending on
external fixture files.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf as fitz  # PyMuPDF (using the non-deprecated import name)
import pytest

from src.pdf.loader import PDFLoadError
from src.pdf.text_extractor import (
    PDFPageText,
    PDFTextExtractionResult,
    PDFWord,
    extract_text_from_pdf,
    extract_text_from_pdf_safe,
)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def single_page_pdf(tmp_path: Path) -> Path:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello world, this is a native text PDF.")
    path = tmp_path / "single_page.pdf"
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def multi_page_pdf(tmp_path: Path) -> Path:
    doc = fitz.open()
    for i in range(3):
        page = doc.new_page()
        page.insert_text((72, 72), f"This is page {i + 1} of a multi-page document.")
    path = tmp_path / "multi_page.pdf"
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def blank_page_pdf(tmp_path: Path) -> Path:
    """A page with no inserted text at all -> empty native text layer."""
    doc = fitz.open()
    doc.new_page()
    path = tmp_path / "blank_page.pdf"
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def invalid_pdf_path(tmp_path: Path) -> Path:
    path = tmp_path / "not_really_a_pdf.pdf"
    path.write_text("This is just plain text, not a PDF file structure.")
    return path


# --------------------------------------------------------------------------
# extract_text_from_pdf
# --------------------------------------------------------------------------


class TestExtractTextFromPDF:
    def test_extracts_text_from_single_page(self, single_page_pdf: Path) -> None:
        result = extract_text_from_pdf(single_page_pdf)

        assert isinstance(result, PDFTextExtractionResult)
        assert len(result.pages) == 1
        assert "Hello world" in result.pages[0].text
        assert result.total_characters == result.pages[0].character_count
        assert result.total_characters > 0

    def test_extracts_all_pages_of_multi_page_document(self, multi_page_pdf: Path) -> None:
        result = extract_text_from_pdf(multi_page_pdf)

        assert len(result.pages) == 3
        assert [p.page_number for p in result.pages] == [1, 2, 3]
        for i, page in enumerate(result.pages, start=1):
            assert f"page {i}" in page.text

    def test_total_characters_is_sum_of_page_characters(self, multi_page_pdf: Path) -> None:
        result = extract_text_from_pdf(multi_page_pdf)
        assert result.total_characters == sum(p.character_count for p in result.pages)

    def test_word_coordinates_are_preserved(self, single_page_pdf: Path) -> None:
        result = extract_text_from_pdf(single_page_pdf)
        page = result.pages[0]

        assert len(page.words) > 0
        for word in page.words:
            assert isinstance(word, PDFWord)
            assert word.text
            # Coordinates should form a valid, non-degenerate box.
            assert word.x1 >= word.x0
            assert word.y1 >= word.y0
            assert isinstance(word.block_number, int)
            assert isinstance(word.line_number, int)
            assert isinstance(word.word_number, int)

    def test_page_dimensions_are_populated(self, single_page_pdf: Path) -> None:
        result = extract_text_from_pdf(single_page_pdf)
        page = result.pages[0]
        assert page.width > 0
        assert page.height > 0

    def test_empty_text_page_returns_empty_text_and_no_words(
        self, blank_page_pdf: Path
    ) -> None:
        result = extract_text_from_pdf(blank_page_pdf)
        page = result.pages[0]
        assert page.text.strip() == ""
        assert page.words == ()
        assert page.character_count == 0

    def test_invalid_pdf_raises_pdf_load_error(self, invalid_pdf_path: Path) -> None:
        with pytest.raises(PDFLoadError):
            extract_text_from_pdf(invalid_pdf_path)

    def test_missing_file_raises_pdf_load_error(self, tmp_path: Path) -> None:
        missing_path = tmp_path / "does_not_exist.pdf"
        with pytest.raises(PDFLoadError):
            extract_text_from_pdf(missing_path)

    def test_document_is_closed_after_extraction(self, multi_page_pdf: Path, monkeypatch) -> None:
        """
        The document must be closed via load_pdf's returned handle even on
        the success path, matching the pattern used throughout src/pdf/.
        """
        closed_flags = []

        import src.pdf.text_extractor as text_extractor_module

        real_load_pdf = text_extractor_module.load_pdf

        def _tracking_load_pdf(path):
            document = real_load_pdf(path)
            original_close = document.close

            def _tracking_close():
                closed_flags.append(True)
                original_close()

            document.close = _tracking_close
            return document

        monkeypatch.setattr(text_extractor_module, "load_pdf", _tracking_load_pdf)

        extract_text_from_pdf(multi_page_pdf)

        assert closed_flags == [True]


# --------------------------------------------------------------------------
# extract_text_from_pdf_safe
# --------------------------------------------------------------------------


class TestExtractTextFromPDFSafe:
    def test_returns_result_on_success(self, single_page_pdf: Path) -> None:
        result = extract_text_from_pdf_safe(single_page_pdf)
        assert result is not None
        assert isinstance(result, PDFTextExtractionResult)

    def test_returns_none_on_invalid_pdf(self, invalid_pdf_path: Path) -> None:
        result = extract_text_from_pdf_safe(invalid_pdf_path)
        assert result is None

    def test_returns_none_on_missing_file(self, tmp_path: Path) -> None:
        missing_path = tmp_path / "does_not_exist.pdf"
        result = extract_text_from_pdf_safe(missing_path)
        assert result is None

    def test_never_raises_on_invalid_pdf(self, invalid_pdf_path: Path) -> None:
        # Should not raise, unlike extract_text_from_pdf.
        extract_text_from_pdf_safe(invalid_pdf_path)
