"""
Tests for ui/app.py, using Streamlit's ``AppTest`` harness
(``streamlit.testing.v1``) to actually execute the script and inspect
what it renders, rather than importing it as a plain module (it's a
top-to-bottom script, not a set of importable functions).

Kept GPU/model-free the same way ``tests/test_unlimited_ocr.py`` is:
a fake ``torch`` module is injected into ``sys.modules`` so the app's
GPU-status banner is exercised deterministically (rather than
depending on whether the machine running these tests happens to have
a real GPU), and ``process_document`` — the single function the UI is
allowed to call for actual pipeline work — is monkeypatched with a
recording fake, so nothing here downloads or runs the real
Unlimited-OCR model. The UI's job is presentation only; these tests
verify that job, not the pipeline itself (which has its own tests).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pymupdf as fitz  # PyMuPDF (non-deprecated import name)
import pytest
from streamlit.testing.v1 import AppTest

from src.ocr.models import BoundingBox, ConfidenceLevel, PageOCRResult, WordResult
from src.pdf.loader import PDFMetadata
from src.pdf.pdf_type_detector import (
    DocumentClassification,
    DocumentType,
    PageClassification,
    PageType,
)
from src.pipeline.document_pipeline import (
    DocumentPipelineError,
    DocumentProcessingResult,
    PageResult,
)

_APP_PATH = str(Path(__file__).resolve().parents[1] / "ui" / "app.py")


# --------------------------------------------------------------------------
# Shared fixtures / fakes
# --------------------------------------------------------------------------


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    doc = fitz.open()
    doc.new_page()
    data = doc.tobytes()
    doc.close()
    return data


class _FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def get_device_name(_index: int = 0) -> str:
        return "Fake Test GPU"


@pytest.fixture
def fake_gpu(monkeypatch):
    """Force the app's GPU-status banner to report a GPU as available."""
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=_FakeCuda()))


@pytest.fixture
def no_gpu(monkeypatch):
    """Force `import torch` to fail, so the app reports no GPU is available."""
    monkeypatch.setitem(sys.modules, "torch", None)


def _fake_ocr_result(page_number: int) -> PageOCRResult:
    regions = [
        {
            "type": "title",
            "text": "Schedule",
            "bounding_box": {"x": 10, "y": 20, "width": 100, "height": 30},
        },
        {
            "type": "table",
            "text": "Mon\tTue\n1\t2",
            "bounding_box": {"x": 10, "y": 60, "width": 200, "height": 80},
            "table_html": "<table><tr><td>Mon</td><td>Tue</td></tr><tr><td>1</td><td>2</td></tr></table>",
        },
    ]
    words = [
        WordResult(text="Schedule", confidence=0.0, bounding_box=BoundingBox(x=10, y=20, width=100, height=30)),
        WordResult(text="Mon\tTue\n1\t2", confidence=0.0, bounding_box=BoundingBox(x=10, y=60, width=200, height=80)),
    ]
    return PageOCRResult(
        page_number=page_number,
        text="Schedule\nMon\tTue\n1\t2",
        language="ara+eng",
        engine="unlimited_ocr",
        words=words,
        mean_confidence=0.0,
        confidence_level=ConfidenceLevel.POOR,
        processing_time_ms=123.4,
        metadata={
            "confidence_available": False,
            "regions": regions,
            "tables": [r for r in regions if "table_html" in r],
        },
    )


def _fake_document_result(pdf_path: str) -> DocumentProcessingResult:
    ocr_page = PageResult(
        page_number=1,
        page_type=PageType.SCANNED,
        character_count=0,
        used_ocr=True,
        ocr_result=_fake_ocr_result(1),
        rendered_image=np.full((300, 300, 3), 255, dtype=np.uint8),
        processed_image=np.full((300, 300, 3), 255, dtype=np.uint8),
    )
    text_page = PageResult(
        page_number=2,
        page_type=PageType.TEXT,
        character_count=42,
        native_text="Hello from a native page.",
        native_quality_good=True,
    )
    metadata = PDFMetadata(filename="sample.pdf", file_path=Path(pdf_path), file_size_bytes=1234, num_pages=2)
    classification = DocumentClassification(
        document_type=DocumentType.MIXED_PDF,
        pages=[
            PageClassification(page_number=1, type=PageType.SCANNED, character_count=0),
            PageClassification(page_number=2, type=PageType.TEXT, character_count=42),
        ],
    )
    return DocumentProcessingResult(
        metadata=metadata,
        classification=classification,
        pages=[ocr_page, text_page],
        ocr_engine_name="unlimited",
        ocr_language="ara+eng",
        ocr_engine_error=None,
        total_processing_time_ms=456.7,
    )


def _run_upload_and_process(at: AppTest, pdf_bytes: bytes) -> AppTest:
    at.run(timeout=20)
    at.file_uploader[0].set_value([("sample.pdf", pdf_bytes, "application/pdf")]).run(timeout=20)
    at.button[0].click().run(timeout=20)
    return at


# --------------------------------------------------------------------------
# App loads cleanly
# --------------------------------------------------------------------------


class TestAppLoads:
    def test_initial_state_renders_without_exception(self):
        at = AppTest.from_file(_APP_PATH)
        at.run(timeout=20)
        assert not at.exception

    def test_initial_state_prompts_for_upload(self):
        at = AppTest.from_file(_APP_PATH)
        at.run(timeout=20)
        assert any("Upload a PDF" in info.value for info in at.info)


# --------------------------------------------------------------------------
# Upload
# --------------------------------------------------------------------------


class TestUpload:
    def test_upload_shows_document_info_without_exception(self, sample_pdf_bytes, fake_gpu):
        at = AppTest.from_file(_APP_PATH)
        at.run(timeout=20)
        at.file_uploader[0].set_value([("sample.pdf", sample_pdf_bytes, "application/pdf")]).run(timeout=20)
        assert not at.exception
        assert any(m.value == "sample.pdf" for m in at.metric)


# --------------------------------------------------------------------------
# Processing calls the pipeline (and only the pipeline)
# --------------------------------------------------------------------------


class TestProcessingCallsThePipeline:
    def test_process_button_calls_process_document_once(self, sample_pdf_bytes, fake_gpu, monkeypatch):
        calls = []

        def _fake_process_document(pdf_path, progress_callback=None):
            calls.append(pdf_path)
            if progress_callback:
                progress_callback("load", "Reading PDF metadata")
                progress_callback("ocr", "Running OCR on page 1")
                progress_callback("done", "Processing complete")
            return _fake_document_result(pdf_path)

        monkeypatch.setattr(
            "src.pipeline.document_pipeline.process_document", _fake_process_document
        )

        at = AppTest.from_file(_APP_PATH)
        _run_upload_and_process(at, sample_pdf_bytes)

        assert not at.exception
        assert len(calls) == 1
        assert calls[0].endswith("sample.pdf")

    def test_processing_is_blocked_without_a_gpu(self, sample_pdf_bytes, no_gpu, monkeypatch):
        calls = []
        monkeypatch.setattr(
            "src.pipeline.document_pipeline.process_document",
            lambda *a, **k: calls.append(1),
        )

        at = AppTest.from_file(_APP_PATH)
        _run_upload_and_process(at, sample_pdf_bytes)

        assert not at.exception
        assert calls == []
        assert any("requires a GPU" in e.value for e in at.error)


# --------------------------------------------------------------------------
# Results are displayed
# --------------------------------------------------------------------------


class TestResultsDisplayed:
    def test_overview_and_tabs_render_without_exception(self, sample_pdf_bytes, fake_gpu, monkeypatch):
        monkeypatch.setattr(
            "src.pipeline.document_pipeline.process_document",
            lambda pdf_path, progress_callback=None: _fake_document_result(pdf_path),
        )

        at = AppTest.from_file(_APP_PATH)
        _run_upload_and_process(at, sample_pdf_bytes)

        assert not at.exception
        assert any("Results" in h.value for h in at.subheader)
        # Overview metrics.
        assert any(m.value == "sample.pdf" for m in at.metric)

    def test_native_and_ocr_text_are_shown(self, sample_pdf_bytes, fake_gpu, monkeypatch):
        monkeypatch.setattr(
            "src.pipeline.document_pipeline.process_document",
            lambda pdf_path, progress_callback=None: _fake_document_result(pdf_path),
        )

        at = AppTest.from_file(_APP_PATH)
        _run_upload_and_process(at, sample_pdf_bytes)

        assert not at.exception
        text_area_values = [ta.value for ta in at.text_area]
        assert any("Hello from a native page." in (v or "") for v in text_area_values) or any(
            "Schedule" in (v or "") for v in text_area_values
        )

    def test_confidence_unavailable_note_shown_for_unlimited_ocr(self, sample_pdf_bytes, fake_gpu, monkeypatch):
        monkeypatch.setattr(
            "src.pipeline.document_pipeline.process_document",
            lambda pdf_path, progress_callback=None: _fake_document_result(pdf_path),
        )

        at = AppTest.from_file(_APP_PATH)
        _run_upload_and_process(at, sample_pdf_bytes)

        assert not at.exception
        captions = [c.value for c in at.caption]
        assert any("does not report a confidence score" in c for c in captions)


# --------------------------------------------------------------------------
# Tables are displayed correctly
# --------------------------------------------------------------------------


class TestTablesDisplayed:
    def test_detected_table_rendered_as_a_dataframe(self, sample_pdf_bytes, fake_gpu, monkeypatch):
        monkeypatch.setattr(
            "src.pipeline.document_pipeline.process_document",
            lambda pdf_path, progress_callback=None: _fake_document_result(pdf_path),
        )

        at = AppTest.from_file(_APP_PATH)
        _run_upload_and_process(at, sample_pdf_bytes)

        assert not at.exception
        assert len(at.dataframe) > 0

    def test_raw_table_html_available_in_an_expander(self, sample_pdf_bytes, fake_gpu, monkeypatch):
        monkeypatch.setattr(
            "src.pipeline.document_pipeline.process_document",
            lambda pdf_path, progress_callback=None: _fake_document_result(pdf_path),
        )

        at = AppTest.from_file(_APP_PATH)
        _run_upload_and_process(at, sample_pdf_bytes)

        assert not at.exception
        code_blocks = [c.value for c in at.code]
        assert any("<table>" in (c or "") for c in code_blocks)

    def test_no_tables_message_when_document_has_none(self, sample_pdf_bytes, fake_gpu, monkeypatch):
        def _no_tables_result(pdf_path, progress_callback=None):
            result = _fake_document_result(pdf_path)
            result.pages[0].ocr_result.metadata["tables"] = []
            return result

        monkeypatch.setattr("src.pipeline.document_pipeline.process_document", _no_tables_result)

        at = AppTest.from_file(_APP_PATH)
        _run_upload_and_process(at, sample_pdf_bytes)

        assert not at.exception
        assert any("No tables were detected" in i.value for i in at.info)


# --------------------------------------------------------------------------
# Downloads work
# --------------------------------------------------------------------------


class TestDownloads:
    def test_text_and_json_download_buttons_present_and_enabled(self, sample_pdf_bytes, fake_gpu, monkeypatch):
        monkeypatch.setattr(
            "src.pipeline.document_pipeline.process_document",
            lambda pdf_path, progress_callback=None: _fake_document_result(pdf_path),
        )

        at = AppTest.from_file(_APP_PATH)
        _run_upload_and_process(at, sample_pdf_bytes)

        assert not at.exception
        labels = [b.label for b in at.download_button]
        assert any("Download Text" in label for label in labels)
        assert any("Download Structured JSON" in label for label in labels)
        assert any("Download Tables" in label for label in labels)

        tables_button = next(b for b in at.download_button if "Download Tables" in b.label)
        assert tables_button.disabled is False  # this fixture's document has a table

    def test_tables_download_disabled_when_no_tables(self, sample_pdf_bytes, fake_gpu, monkeypatch):
        def _no_tables_result(pdf_path, progress_callback=None):
            result = _fake_document_result(pdf_path)
            result.pages[0].ocr_result.metadata["tables"] = []
            return result

        monkeypatch.setattr("src.pipeline.document_pipeline.process_document", _no_tables_result)

        at = AppTest.from_file(_APP_PATH)
        _run_upload_and_process(at, sample_pdf_bytes)

        tables_button = next(b for b in at.download_button if "Download Tables" in b.label)
        assert tables_button.disabled is True


# --------------------------------------------------------------------------
# Errors are handled cleanly
# --------------------------------------------------------------------------


class TestErrorHandling:
    def test_document_pipeline_error_shows_its_own_message_with_no_traceback(
        self, sample_pdf_bytes, fake_gpu, monkeypatch
    ):
        monkeypatch.setattr(
            "src.pipeline.document_pipeline.process_document",
            lambda *a, **k: (_ for _ in ()).throw(DocumentPipelineError("could not read this PDF")),
        )

        at = AppTest.from_file(_APP_PATH)
        _run_upload_and_process(at, sample_pdf_bytes)

        assert not at.exception
        assert any("could not read this PDF" in e.value for e in at.error)
        assert len(at.expander) == 0

    def test_unexpected_error_shows_friendly_message_with_details_hidden_in_expander(
        self, sample_pdf_bytes, fake_gpu, monkeypatch
    ):
        monkeypatch.setattr(
            "src.pipeline.document_pipeline.process_document",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("CUDA out of memory: boom")),
        )

        at = AppTest.from_file(_APP_PATH)
        _run_upload_and_process(at, sample_pdf_bytes)

        assert not at.exception
        error_messages = [e.value for e in at.error]
        assert any("Something went wrong" in msg for msg in error_messages)
        # The raw exception text must not leak into the main error banner...
        assert not any("CUDA out of memory" in msg for msg in error_messages)
        # ...but must still be available for debugging, tucked away.
        assert len(at.expander) >= 1
        code_blocks = [c.value for c in at.code]
        assert any("CUDA out of memory" in (c or "") for c in code_blocks)
