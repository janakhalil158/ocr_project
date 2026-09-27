"""
Real, end-to-end tests for Baidu Unlimited-OCR: actually loads
``baidu/Unlimited-OCR`` and runs inference on a real page image.

NOT part of the default test run -- every test in this file is marked
``@pytest.mark.gpu`` and skipped unless explicitly opted into, because
running them for real requires:

  * torch + transformers (+ this project's other requirements) installed
  * a CUDA-capable GPU (see src/ocr/unlimited_ocr.py -- this model is
    not the supported CPU-inference path)
  * network access (or a pre-populated local cache) to download the
    ~6.67 GB model weights the first time

Opt in explicitly with:

    RUN_UNLIMITED_OCR_GPU_TESTS=1 pytest tests/test_unlimited_ocr_integration.py

On a machine that meets the requirements above, dropping a real PDF
into ``data/sample/pdfs/`` gives these tests something to run against;
without one, they skip individually with a clear reason rather than
failing.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_RUN_GPU_TESTS = os.environ.get("RUN_UNLIMITED_OCR_GPU_TESTS", "").strip().lower() in (
    "1",
    "true",
    "yes",
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not _RUN_GPU_TESTS,
        reason=(
            "Real Unlimited-OCR GPU/model integration tests are skipped by "
            "default (they need torch/transformers, a CUDA GPU, and a "
            "~6.67 GB model download). Set RUN_UNLIMITED_OCR_GPU_TESTS=1 to "
            "run them."
        ),
    ),
]

_SAMPLE_PDF_DIR = Path(__file__).resolve().parents[1] / "data" / "sample" / "pdfs"


def _first_sample_pdf() -> Path:
    pdfs = sorted(_SAMPLE_PDF_DIR.glob("*.pdf")) if _SAMPLE_PDF_DIR.is_dir() else []
    if not pdfs:
        pytest.skip(f"No sample PDF found under {_SAMPLE_PDF_DIR} to run against.")
    return pdfs[0]


class TestUnlimitedOCRRealModel:
    def test_model_loads_on_gpu(self):
        from src.ocr.unlimited_ocr import UnlimitedOCREngine

        engine = UnlimitedOCREngine()
        model, tokenizer = engine._get_model_and_tokenizer()
        assert model is not None
        assert tokenizer is not None

    def test_recognize_raw_produces_regions_on_a_real_page(self):
        from src.ocr.unlimited_ocr import UnlimitedOCREngine
        from src.pdf.page_renderer import render_page_to_array

        engine = UnlimitedOCREngine()
        image = render_page_to_array(str(_first_sample_pdf()), 1, dpi=200)
        data = engine.recognize_raw(image, language="ara+eng")

        assert "region_type" in data
        assert len(data["text"]) > 0

    def test_extract_text_end_to_end_through_orchestration(self):
        from src.ocr.ocr import extract_text
        from src.ocr.unlimited_ocr import UnlimitedOCREngine
        from src.pdf.page_renderer import render_page_to_array

        engine = UnlimitedOCREngine()
        image = render_page_to_array(str(_first_sample_pdf()), 1, dpi=200)
        result = extract_text(image, engine=engine)

        assert result.engine == "unlimited_ocr"
        assert result.metadata["confidence_available"] is False

    def test_document_pipeline_processes_a_real_pdf(self):
        from src.pipeline.document_pipeline import process_document

        result = process_document(str(_first_sample_pdf()))
        assert result.pages
