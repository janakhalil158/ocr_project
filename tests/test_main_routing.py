"""
Integration tests for src/main.py's per-page-type routing.

Verifies the behavior described in the architecture:

    TEXT_PDF pages    -> native extraction only, OCR engine never built
    SCANNED_PDF pages -> unchanged quality -> preprocessing -> OCR path

Builds small synthetic PDFs with PyMuPDF (matching tests/test_pdf.py's
approach) rather than depending on external fixture files.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf as fitz  # PyMuPDF (using the non-deprecated import name)
import pytest

import src.ocr.factory as ocr_factory
from src.main import print_report


@pytest.fixture
def text_pdf_path(tmp_path: Path) -> Path:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        "This is a native, digitally-authored PDF page with real text content.",
    )
    path = tmp_path / "text_document.pdf"
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def scanned_pdf_path(tmp_path: Path) -> Path:
    # No inserted text -> classified as SCANNED (no machine-readable layer).
    doc = fitz.open()
    doc.new_page()
    path = tmp_path / "scanned_document.pdf"
    doc.save(str(path))
    doc.close()
    return path


class TestTextPDFRouting:
    def test_native_extraction_output_is_printed(
        self, text_pdf_path: Path, capsys
    ) -> None:
        exit_code = print_report(str(text_pdf_path))
        captured = capsys.readouterr()

        assert exit_code == 0
        assert "NATIVE TEXT EXTRACTION" in captured.out
        assert "TABLE / LAYOUT ANALYSIS" in captured.out
        assert "digitally-authored PDF page" in captured.out

    def test_ocr_engine_is_never_constructed_for_pure_text_pdf(
        self, text_pdf_path: Path, monkeypatch
    ) -> None:
        calls = []
        original_get_ocr_engine = ocr_factory.get_ocr_engine

        def _tracking_get_ocr_engine(*args, **kwargs):
            calls.append((args, kwargs))
            return original_get_ocr_engine(*args, **kwargs)

        monkeypatch.setattr(ocr_factory, "get_ocr_engine", _tracking_get_ocr_engine)

        print_report(str(text_pdf_path))

        assert calls == [], (
            "get_ocr_engine() was called while processing a pure TEXT_PDF; "
            "OCR must not be initialized for native-text pages."
        )

    def test_image_quality_assessment_is_not_printed_for_text_pdf(
        self, text_pdf_path: Path, capsys
    ) -> None:
        print_report(str(text_pdf_path))
        captured = capsys.readouterr()
        assert "IMAGE QUALITY ASSESSMENT" not in captured.out


class _FakeScannedPathEngine:
    """
    Minimal stand-in :class:`~src.ocr.base.OCREngine` used only to keep
    this test offline/deterministic. This test's purpose is confirming
    the SCANNED_PDF execution path (quality -> preprocessing -> OCR)
    still runs unchanged; it isn't about which engine is configured,
    and must not construct a real Unlimited-OCR engine (which would
    try to download/load a 6+ GB model on first use).
    """

    name = "fake"

    def recognize_raw(self, image, language):
        return {
            "text": ["FAKE"],
            "conf": [90.0],
            "left": [0],
            "top": [0],
            "width": [10],
            "height": [10],
            "block_num": [1],
            "par_num": [1],
            "line_num": [1],
        }


class TestScannedPDFRouting:
    def test_scanned_path_still_runs_quality_preprocessing_ocr(
        self, scanned_pdf_path: Path, capsys, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            ocr_factory, "get_ocr_engine", lambda *a, **k: _FakeScannedPathEngine()
        )

        exit_code = print_report(str(scanned_pdf_path))
        captured = capsys.readouterr()

        assert exit_code == 0
        assert "IMAGE QUALITY ASSESSMENT" in captured.out
        assert "PREPROCESSING" in captured.out
        assert "OCR" in captured.out
        # Scanned-only documents should not print the native-extraction
        # section at all.
        assert "NATIVE TEXT EXTRACTION" not in captured.out
