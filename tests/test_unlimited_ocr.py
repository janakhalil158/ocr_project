"""
Tests for src/ocr/unlimited_ocr.py: Baidu Unlimited-OCR's adapter.

GPU/model-free by design: every test here either exercises pure
parsing functions (``parse_detections``, ``html_table_to_text``) or
monkeypatches the model-calling seam (``_run_model``, or the fake
``torch``/``transformers`` modules used to build one) so nothing in
this file downloads or runs the real 6+ GB model, and none of it
requires torch/transformers to be installed at all. Real end-to-end
GPU inference is covered separately in
``tests/test_unlimited_ocr_integration.py``, skipped by default.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from src.ocr.models import OCREngineNotAvailableError
from src.ocr.unlimited_ocr import (
    COORDINATE_GRID_SIZE,
    UnlimitedOCREngine,
    _import_transformers,
    _scale_coordinate,
    html_table_to_text,
    parse_detections,
)

# --------------------------------------------------------------------------
# parse_detections()
# --------------------------------------------------------------------------


class TestParseDetections:
    def test_single_region_parsed(self):
        raw = "<|det|>title [100, 200, 300, 400]<|/det|>Hello World"
        regions = parse_detections(raw, image_height=1000, image_width=1000)
        assert len(regions) == 1
        region = regions[0]
        assert region["type"] == "title"
        assert region["text"] == "Hello World"
        assert region["table_html"] is None

    def test_reading_order_preserved(self):
        raw = (
            "<|det|>header [0, 0, 100, 50]<|/det|>Header text"
            "<|det|>title [0, 60, 200, 100]<|/det|>Title text"
            "<|det|>text [0, 110, 300, 200]<|/det|>Body text"
        )
        regions = parse_detections(raw, image_height=1000, image_width=1000)
        assert [r["type"] for r in regions] == ["header", "title", "text"]
        assert [r["text"] for r in regions] == ["Header text", "Title text", "Body text"]

    def test_open_ended_region_types_passed_through(self):
        # Region types are not mapped onto a fixed enum -- whatever the
        # model reports must survive verbatim (lowercased).
        raw = "<|det|>page_number [0, 900, 100, 950]<|/det|>Page 1 of 1"
        regions = parse_detections(raw, image_height=1000, image_width=1000)
        assert regions[0]["type"] == "page_number"

    def test_non_text_marker_becomes_empty_text_but_region_kept(self):
        raw = "<|det|>header [900, 24, 938, 81]<|/det|>[Non-Text]"
        regions = parse_detections(raw, image_height=1000, image_width=1000)
        assert len(regions) == 1
        assert regions[0]["type"] == "header"
        assert regions[0]["text"] == ""
        # The literal marker is never surfaced as text, but the box is real.
        assert regions[0]["width"] > 0 and regions[0]["height"] > 0

    def test_table_html_preserved_and_plain_text_generated(self):
        raw = (
            "<|det|>table [0, 0, 500, 500]<|/det|>"
            "<table><tr><td>Name</td><td>Score</td></tr>"
            "<tr><td>Alice</td><td>90</td></tr></table>"
        )
        regions = parse_detections(raw, image_height=1000, image_width=1000)
        region = regions[0]
        assert region["type"] == "table"
        assert region["table_html"].startswith("<table>")
        assert "Name" in region["text"] and "Score" in region["text"]
        assert "Alice" in region["text"] and "90" in region["text"]
        # The region's normal-output text is plain text, not raw markup.
        assert "<table>" not in region["text"]

    def test_no_detections_returns_empty_list(self):
        assert parse_detections("plain text, no det tags", image_height=1000, image_width=1000) == []
        assert parse_detections("", image_height=1000, image_width=1000) == []

    def test_bounding_box_rescaled_from_normalized_grid_to_pixels(self):
        # x1=0,y1=0,x2=500,y2=250 on a 0-1000 grid, against a real
        # 2000(w) x 1000(h) image -> half the width, a quarter of the
        # height. This is the core "not raw pixel coordinates" check.
        raw = "<|det|>text [0, 0, 500, 250]<|/det|>Some text"
        regions = parse_detections(raw, image_height=1000, image_width=2000)
        region = regions[0]
        assert region["left"] == 0
        assert region["top"] == 0
        assert region["width"] == 1000  # 500/1000 * 2000
        assert region["height"] == 250  # 250/1000 * 1000

    def test_bounding_box_rescaling_uses_full_grid_extent(self):
        raw = "<|det|>text [0, 0, 1000, 1000]<|/det|>Full page"
        regions = parse_detections(raw, image_height=800, image_width=600)
        region = regions[0]
        assert region["left"] == 0 and region["top"] == 0
        assert region["width"] == 600
        assert region["height"] == 800

    def test_scale_coordinate_handles_zero_pixel_size(self):
        assert _scale_coordinate(500, COORDINATE_GRID_SIZE, 0) == 0

    def test_coordinates_never_negative_even_if_reversed(self):
        # x1 > x2 / y1 > y2 shouldn't happen in practice, but a
        # reversed box must never produce a negative width/height.
        raw = "<|det|>text [300, 300, 100, 100]<|/det|>Odd box"
        regions = parse_detections(raw, image_height=1000, image_width=1000)
        region = regions[0]
        assert region["width"] >= 0 and region["height"] >= 0


# --------------------------------------------------------------------------
# html_table_to_text()
# --------------------------------------------------------------------------


class TestHtmlTableToText:
    def test_simple_table_rendered_as_tab_separated_rows(self):
        html = "<table><tr><td>A</td><td>B</td></tr><tr><td>1</td><td>2</td></tr></table>"
        assert html_table_to_text(html) == "A\tB\n1\t2"

    def test_th_cells_supported(self):
        html = "<table><tr><th>Name</th><th>Age</th></tr><tr><td>Sam</td><td>30</td></tr></table>"
        text = html_table_to_text(html)
        assert "Name" in text and "Sam" in text

    def test_malformed_html_falls_back_without_raising(self):
        text = html_table_to_text("<table><tr><td>Unclosed")
        assert isinstance(text, str)


# --------------------------------------------------------------------------
# UnlimitedOCREngine.recognize_raw() -- with _run_model stubbed out
# --------------------------------------------------------------------------


class _StubbedEngine(UnlimitedOCREngine):
    """UnlimitedOCREngine with model inference replaced by canned text."""

    def __init__(self, canned_raw_output: str, **kwargs):
        super().__init__(**kwargs)
        self._canned_raw_output = canned_raw_output

    def _run_model(self, image: np.ndarray) -> str:  # noqa: D401 -- test double
        return self._canned_raw_output


class TestRecognizeRaw:
    _SAMPLE_OUTPUT = (
        "<|det|>title [100, 50, 900, 120]<|/det|>Schedule\n"
        "<|det|>text [100, 150, 900, 200]<|/det|>Student Name\n"
        "<|det|>table [50, 250, 950, 700]<|/det|><table><tr><td>Mon</td><td>Tue</td></tr></table>\n"
        "<|det|>footer [700, 970, 950, 990]<|/det|>November 21, 2025"
    )

    def test_returns_required_tesseract_shaped_keys(self):
        engine = _StubbedEngine(self._SAMPLE_OUTPUT)
        data = engine.recognize_raw(np.full((1000, 1000, 3), 255, dtype=np.uint8), "ara+eng")
        for key in ("text", "conf", "left", "top", "width", "height", "block_num", "par_num", "line_num"):
            assert key in data
            assert len(data[key]) == 4

    def test_confidence_is_always_the_documented_zero_placeholder(self):
        engine = _StubbedEngine(self._SAMPLE_OUTPUT)
        data = engine.recognize_raw(np.full((1000, 1000, 3), 255, dtype=np.uint8), "ara+eng")
        assert data["conf"] == [0.0, 0.0, 0.0, 0.0]
        assert data["engine_reports_confidence"] is False

    def test_region_type_and_table_html_parallel_arrays_present(self):
        engine = _StubbedEngine(self._SAMPLE_OUTPUT)
        data = engine.recognize_raw(np.full((1000, 1000, 3), 255, dtype=np.uint8), "ara+eng")
        assert data["region_type"] == ["title", "text", "table", "footer"]
        assert data["table_html"][2] is not None
        assert data["table_html"][0] is None

    def test_block_par_line_numbering_matches_region_level_convention(self):
        engine = _StubbedEngine(self._SAMPLE_OUTPUT)
        data = engine.recognize_raw(np.full((1000, 1000, 3), 255, dtype=np.uint8), "ara+eng")
        assert data["block_num"] == [1, 1, 1, 1]
        assert data["par_num"] == [1, 1, 1, 1]
        assert data["line_num"] == [1, 2, 3, 4]

    def test_grayscale_image_dimensions_handled(self):
        engine = _StubbedEngine("<|det|>text [0, 0, 500, 500]<|/det|>Hi")
        data = engine.recognize_raw(np.full((400, 800), 255, dtype=np.uint8), "eng")
        # width(800) * 500/1000 = 400 ; height(400) * 500/1000 = 200
        assert data["width"][0] == 400
        assert data["height"][0] == 200

    def test_empty_model_output_returns_empty_rows(self):
        engine = _StubbedEngine("")
        data = engine.recognize_raw(np.full((100, 100, 3), 255, dtype=np.uint8), "eng")
        assert data["text"] == []
        assert data["region_type"] == []


# --------------------------------------------------------------------------
# Engine construction is cheap/lazy: no transformers/torch import (and
# no model download) until recognize_raw actually needs the model.
# --------------------------------------------------------------------------


class TestLazyConstruction:
    def test_construction_succeeds_without_transformers_installed(self):
        # transformers genuinely isn't installed in this test
        # environment -- constructing the engine must still succeed,
        # proving model loading truly hasn't happened yet.
        engine = UnlimitedOCREngine()
        assert engine.model_name == "baidu/Unlimited-OCR"
        assert engine.device == "cuda"

    def test_import_transformers_raises_controlled_error_when_missing(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "transformers", None)
        with pytest.raises(OCREngineNotAvailableError):
            _import_transformers()


# --------------------------------------------------------------------------
# Model loading + process-wide caching (fake torch/transformers, no
# real download or GPU required)
# --------------------------------------------------------------------------


def _make_fake_transformers_and_torch():
    """Build minimal fake ``transformers``/``torch`` modules for cache tests."""

    class _FakeModel:
        def eval(self):
            return self

        def to(self, *_args, **_kwargs):
            return self

    class _FakeTokenizer:
        pass

    build_calls = {"count": 0}

    class _FakeAutoModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            build_calls["count"] += 1
            return _FakeModel()

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return _FakeTokenizer()

    fake_transformers = types.SimpleNamespace(AutoModel=_FakeAutoModel, AutoTokenizer=_FakeAutoTokenizer)
    fake_torch = types.SimpleNamespace(
        bfloat16="bfloat16-marker", float16="float16-marker", float32="float32-marker"
    )
    return fake_transformers, fake_torch, build_calls


class TestModelCaching:
    def setup_method(self):
        # Each test starts from a clean process-wide cache so tests
        # don't leak loaded "models" into each other.
        from src.ocr import unlimited_ocr

        unlimited_ocr._MODEL_CACHE.clear()

    def test_model_is_loaded_once_and_reused_across_engine_instances(self, monkeypatch):
        fake_transformers, fake_torch, build_calls = _make_fake_transformers_and_torch()
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        engine_a = UnlimitedOCREngine(transformers_module=fake_transformers)
        engine_b = UnlimitedOCREngine(transformers_module=fake_transformers)

        model_a, _ = engine_a._get_model_and_tokenizer()
        model_b, _ = engine_b._get_model_and_tokenizer()

        assert build_calls["count"] == 1
        assert model_a is model_b

    def test_different_device_gets_its_own_cache_entry(self, monkeypatch):
        fake_transformers, fake_torch, build_calls = _make_fake_transformers_and_torch()
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        engine_gpu = UnlimitedOCREngine(device="cuda", transformers_module=fake_transformers)
        engine_cpu = UnlimitedOCREngine(device="cpu", transformers_module=fake_transformers)

        engine_gpu._get_model_and_tokenizer()
        engine_cpu._get_model_and_tokenizer()

        assert build_calls["count"] == 2

    def test_model_load_failure_becomes_controlled_error(self, monkeypatch):
        class _FailingAutoModel:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                raise RuntimeError("out of memory")

        fake_transformers = types.SimpleNamespace(
            AutoModel=_FailingAutoModel,
            AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: object()),
        )
        _, fake_torch, _ = _make_fake_transformers_and_torch()
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        engine = UnlimitedOCREngine(transformers_module=fake_transformers)
        with pytest.raises(OCREngineNotAvailableError):
            engine._get_model_and_tokenizer()
