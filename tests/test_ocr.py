"""
Tests for Phase 4: the shared OCR orchestration layer (src/ocr/ocr.py).

Deliberately engine-agnostic and GPU/model-free: every test here drives
`extract_text()`/`extract_text_safe()` through a small in-process
`FakeOCREngine` that returns a canned, Tesseract-shaped raw dict (the
same contract `OCREngine.recognize_raw()` documents), so this file
never depends on Tesseract, EasyOCR, or the real Unlimited-OCR model
being installed/downloaded. That's the module's own design goal
(stated in its docstring): "this module deliberately contains no
engine-specific code."

Unlimited-OCR-specific behavior (parsing its `<|det|>` output, table
HTML extraction, coordinate rescaling) is covered separately in
`tests/test_unlimited_ocr.py` (also GPU/model-free). Real model-loading
and end-to-end GPU inference are covered in
`tests/test_unlimited_ocr_integration.py`, which is skipped by default.

Note on scope changes from the previous (Tesseract-only) version of
this file:
  * Tests that asserted real OCR actually read specific text out of a
    rendered image (e.g. "recognizes HELLO WORLD") no longer apply --
    there's no real recognition happening at this layer. Equivalent
    intent (language is carried through / passed to the engine, line
    reconstruction groups words correctly, etc.) is kept via canned
    data.
  * The Arabic-font-rendering test class is removed outright: it was a
    Tesseract Arabic-language-pack capability test, not an
    orchestration test, and testing a fake engine with Arabic text
    would exercise nothing this file doesn't already cover (data is
    handled as opaque strings regardless of script). Genuine
    multilingual accuracy is a GPU integration concern.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

from src.ocr.base import OCREngine
from src.ocr.models import ConfidenceLevel, ConfidenceThresholds, confidence_level
from src.ocr.ocr import (
    BoundingBox,
    OCRConfig,
    OCRError,
    PageOCRResult,
    WordResult,
    extract_text,
    extract_text_safe,
)

# --------------------------------------------------------------------------
# Synthetic image builders
# --------------------------------------------------------------------------
#
# No real OCR happens at this layer, so these just need to be
# well-formed uint8 arrays of the right shape/dtype -- not images that
# actually render legible text.


def _make_image(width: int = 200, height: int = 100, channels: Optional[int] = 3) -> np.ndarray:
    shape = (height, width, channels) if channels else (height, width)
    return np.full(shape, 255, dtype=np.uint8)


# --------------------------------------------------------------------------
# FakeOCREngine: the shared test double for every test in this file
# --------------------------------------------------------------------------


def _canned_raw_data(
    words_with_boxes: List[Tuple[str, float, Tuple[int, int, int, int]]],
    line_nums: Optional[List[int]] = None,
    region_type: Optional[List[str]] = None,
    table_html: Optional[List[Optional[str]]] = None,
    engine_reports_confidence: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Build a minimal Tesseract-shaped raw dict, as any
    ``OCREngine.recognize_raw()`` must return.

    ``line_nums`` lets a test group multiple rows onto the same
    reconstructed line (all default to their own line, i.e. one row
    per line, mirroring the "one detection = one row" convention
    EasyOCR's and Unlimited-OCR's adapters both use). The optional
    ``region_type``/``table_html``/``engine_reports_confidence`` mirror
    the *extra*, optional keys Unlimited-OCR's adapter adds -- used
    here to test that extract_text() folds them into
    ``PageOCRResult.metadata`` regardless of which engine supplied
    them.
    """
    data: Dict[str, Any] = {
        "text": [],
        "conf": [],
        "left": [],
        "top": [],
        "width": [],
        "height": [],
        "block_num": [],
        "par_num": [],
        "line_num": [],
    }
    for i, (text, conf, box) in enumerate(words_with_boxes, start=1):
        data["text"].append(text)
        data["conf"].append(conf)
        data["left"].append(box[0])
        data["top"].append(box[1])
        data["width"].append(box[2])
        data["height"].append(box[3])
        data["block_num"].append(1)
        data["par_num"].append(1)
        data["line_num"].append(line_nums[i - 1] if line_nums else i)

    if region_type is not None:
        data["region_type"] = region_type
    if table_html is not None:
        data["table_html"] = table_html
    if engine_reports_confidence is not None:
        data["engine_reports_confidence"] = engine_reports_confidence

    return data


class FakeOCREngine(OCREngine):
    """
    A minimal, in-process :class:`~src.ocr.base.OCREngine` that returns
    a canned raw dict instead of running any real recognition. Used by
    every test in this file so the shared orchestration in
    :func:`~src.ocr.ocr.extract_text` can be exercised without
    Tesseract, EasyOCR, or Unlimited-OCR/torch/a GPU being available.
    """

    name = "fake"

    def __init__(self, raw_data: Optional[Dict[str, Any]] = None, error: Optional[Exception] = None):
        self._raw_data = raw_data if raw_data is not None else _canned_raw_data([])
        self._error = error
        self.calls: List[Tuple[np.ndarray, str]] = []

    def recognize_raw(self, image: np.ndarray, language: str) -> Dict[str, Any]:
        self.calls.append((image, language))
        if self._error is not None:
            raise self._error
        return self._raw_data


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------


class TestInputValidation:
    def test_none_image_raises(self):
        with pytest.raises(OCRError):
            extract_text(None, engine=FakeOCREngine())

    def test_wrong_type_raises(self):
        with pytest.raises(OCRError):
            extract_text([[1, 2], [3, 4]], engine=FakeOCREngine())

    def test_wrong_dtype_raises(self):
        image = np.full((100, 100, 3), 255, dtype=np.float32)
        with pytest.raises(OCRError):
            extract_text(image, engine=FakeOCREngine())

    def test_too_small_image_raises(self):
        image = np.full((2, 2, 3), 255, dtype=np.uint8)
        with pytest.raises(OCRError):
            extract_text(image, engine=FakeOCREngine())

    def test_rgb_input_supported(self):
        engine = FakeOCREngine(_canned_raw_data([("HELLO", 90.0, (0, 0, 40, 20))]))
        result = extract_text(_make_image(channels=3), engine=engine)
        assert result.text == "HELLO"

    def test_grayscale_input_supported(self):
        engine = FakeOCREngine(_canned_raw_data([("HELLO", 90.0, (0, 0, 40, 20))]))
        result = extract_text(_make_image(channels=None), engine=engine)
        assert result.text == "HELLO"


# --------------------------------------------------------------------------
# Text recognition / line reconstruction
# --------------------------------------------------------------------------


class TestTextRecognition:
    def test_known_text_recognized(self):
        engine = FakeOCREngine(
            _canned_raw_data([("HELLO", 90.0, (0, 0, 40, 20)), ("WORLD", 90.0, (50, 0, 40, 20))])
        )
        result = extract_text(_make_image(), engine=engine)
        assert "HELLO" in result.text
        assert "WORLD" in result.text

    def test_multiple_words_produce_multiple_word_results(self):
        engine = FakeOCREngine(
            _canned_raw_data([("HELLO", 90.0, (0, 0, 40, 20)), ("WORLD", 90.0, (50, 0, 40, 20))])
        )
        result = extract_text(_make_image(), engine=engine)
        assert len(result.words) == 2

    def test_multiple_lines_preserved_in_text(self):
        engine = FakeOCREngine(
            _canned_raw_data(
                [("HELLO", 90.0, (0, 0, 40, 20)), ("SECOND", 90.0, (0, 30, 40, 20))],
                line_nums=[1, 2],
            )
        )
        result = extract_text(_make_image(), engine=engine)
        lines = result.text.split("\n")
        assert len(lines) == 2
        assert lines[0] == "HELLO"
        assert lines[1] == "SECOND"

    def test_blank_page_returns_empty_text(self):
        engine = FakeOCREngine(_canned_raw_data([]))
        result = extract_text(_make_image(), engine=engine)
        assert result.text == ""


# --------------------------------------------------------------------------
# Result structure
# --------------------------------------------------------------------------


class TestResultStructure:
    def test_result_is_page_ocr_result_instance(self):
        engine = FakeOCREngine(_canned_raw_data([("X", 90.0, (0, 0, 10, 10))]))
        result = extract_text(_make_image(), engine=engine)
        assert isinstance(result, PageOCRResult)

    def test_words_are_word_result_instances(self):
        engine = FakeOCREngine(_canned_raw_data([("X", 90.0, (0, 0, 10, 10))]))
        result = extract_text(_make_image(), engine=engine)
        assert all(isinstance(w, WordResult) for w in result.words)

    def test_bounding_boxes_are_bounding_box_instances(self):
        engine = FakeOCREngine(_canned_raw_data([("X", 90.0, (0, 0, 10, 10))]))
        result = extract_text(_make_image(), engine=engine)
        assert all(isinstance(w.bounding_box, BoundingBox) for w in result.words)

    def test_page_number_carried_through(self):
        engine = FakeOCREngine(_canned_raw_data([("X", 90.0, (0, 0, 10, 10))]))
        result = extract_text(_make_image(), page_number=42, engine=engine)
        assert result.page_number == 42

    def test_to_dict_matches_expected_shape(self):
        engine = FakeOCREngine(_canned_raw_data([("X", 90.0, (0, 0, 10, 10))]))
        result = extract_text(_make_image(), engine=engine)
        payload = result.to_dict()
        for key in ("page_number", "text", "word_count", "mean_confidence", "words"):
            assert key in payload
        for word_payload in payload["words"]:
            for key in ("text", "confidence", "bounding_box"):
                assert key in word_payload
            for key in ("x", "y", "width", "height"):
                assert key in word_payload["bounding_box"]


# --------------------------------------------------------------------------
# Confidence
# --------------------------------------------------------------------------


class TestConfidence:
    def test_word_confidence_in_valid_range(self):
        engine = FakeOCREngine(
            _canned_raw_data([("HELLO", 82.0, (0, 0, 10, 10)), ("WORLD", 77.0, (20, 0, 10, 10))])
        )
        result = extract_text(_make_image(), engine=engine)
        assert result.words
        for word in result.words:
            assert 0.0 <= word.confidence <= 100.0

    def test_mean_confidence_matches_manual_average(self):
        engine = FakeOCREngine(
            _canned_raw_data([("HELLO", 82.0, (0, 0, 10, 10)), ("WORLD", 60.0, (20, 0, 10, 10))])
        )
        result = extract_text(_make_image(), engine=engine)
        expected = sum(w.confidence for w in result.words) / len(result.words)
        assert result.mean_confidence == pytest.approx(expected)

    def test_blank_page_has_zero_mean_confidence_and_no_words(self):
        engine = FakeOCREngine(_canned_raw_data([]))
        result = extract_text(_make_image(), engine=engine)
        assert result.words == []
        assert result.mean_confidence == 0.0

    def test_min_confidence_filters_out_words(self):
        engine = FakeOCREngine(
            _canned_raw_data([("HELLO", 90.0, (0, 0, 10, 10)), ("WORLD", 95.0, (20, 0, 10, 10))])
        )
        result = extract_text(_make_image(), config=OCRConfig(min_confidence=101), engine=engine)
        assert result.words == []
        assert result.text == ""
        assert result.mean_confidence == 0.0

    def test_placeholder_zero_confidence_is_kept_at_default_threshold(self):
        # Mirrors Unlimited-OCR's real behavior: every row reports the
        # documented 0.0 placeholder (see src/ocr/unlimited_ocr.py),
        # never a fabricated positive number. At the default
        # min_confidence=0.0 this must NOT filter the text out.
        engine = FakeOCREngine(
            _canned_raw_data(
                [("SOME TEXT", 0.0, (0, 0, 10, 10))],
                engine_reports_confidence=False,
            )
        )
        result = extract_text(_make_image(), engine=engine)
        assert result.text == "SOME TEXT"
        assert result.mean_confidence == 0.0
        assert result.metadata["confidence_available"] is False


# --------------------------------------------------------------------------
# Bounding boxes
# --------------------------------------------------------------------------


class TestBoundingBoxes:
    def test_boxes_exist_for_every_word(self):
        engine = FakeOCREngine(
            _canned_raw_data([("HELLO", 90.0, (0, 0, 10, 10)), ("WORLD", 90.0, (20, 0, 10, 10))])
        )
        result = extract_text(_make_image(), engine=engine)
        assert len(result.words) == 2
        assert all(w.bounding_box is not None for w in result.words)

    def test_coordinates_are_non_negative(self):
        engine = FakeOCREngine(_canned_raw_data([("X", 90.0, (5, 7, 10, 10))]))
        result = extract_text(_make_image(), engine=engine)
        for word in result.words:
            box = word.bounding_box
            assert box.x >= 0 and box.y >= 0

    def test_width_and_height_are_positive(self):
        engine = FakeOCREngine(_canned_raw_data([("X", 90.0, (0, 0, 10, 10))]))
        result = extract_text(_make_image(), engine=engine)
        for word in result.words:
            box = word.bounding_box
            assert box.width > 0 and box.height > 0

    def test_second_word_positioned_right_of_first(self):
        engine = FakeOCREngine(
            _canned_raw_data([("HELLO", 90.0, (0, 0, 40, 20)), ("WORLD", 90.0, (50, 0, 40, 20))])
        )
        result = extract_text(_make_image(), engine=engine)
        hello, world = result.words[0], result.words[1]
        assert world.bounding_box.x > hello.bounding_box.x


# --------------------------------------------------------------------------
# Region / table metadata (engine-agnostic mechanism; see
# src/ocr/unlimited_ocr.py for the engine that actually populates it)
# --------------------------------------------------------------------------


class TestRegionMetadata:
    def test_region_type_and_bounding_boxes_survive_into_metadata(self):
        engine = FakeOCREngine(
            _canned_raw_data(
                [("Invoice", 0.0, (10, 20, 100, 30)), ("Total: $50", 0.0, (10, 60, 100, 20))],
                region_type=["title", "text"],
                table_html=[None, None],
                engine_reports_confidence=False,
            )
        )
        result = extract_text(_make_image(), engine=engine)
        regions = result.metadata["regions"]
        assert [r["type"] for r in regions] == ["title", "text"]
        assert regions[0]["bounding_box"] == {"x": 10, "y": 20, "width": 100, "height": 30}
        assert regions[0]["text"] == "Invoice"

    def test_table_html_preserved_in_metadata(self):
        html = "<table><tr><td>A</td><td>B</td></tr></table>"
        engine = FakeOCREngine(
            _canned_raw_data(
                [("A\tB", 0.0, (0, 0, 100, 40))],
                region_type=["table"],
                table_html=[html],
                engine_reports_confidence=False,
            )
        )
        result = extract_text(_make_image(), engine=engine)
        assert result.metadata["tables"] == [
            {
                "type": "table",
                "text": "A\tB",
                "bounding_box": {"x": 0, "y": 0, "width": 100, "height": 40},
                "table_html": html,
            }
        ]

    def test_non_text_region_kept_in_metadata_but_excluded_from_words(self):
        # A [Non-Text] detection is represented as an empty-text region
        # by the engine adapter (see src/ocr/unlimited_ocr.py) -- it
        # must still appear in metadata["regions"], but never as a word
        # or in the reconstructed text.
        engine = FakeOCREngine(
            _canned_raw_data(
                [("", 0.0, (0, 0, 50, 50)), ("Real text", 0.0, (0, 60, 50, 20))],
                region_type=["header", "text"],
                table_html=[None, None],
                engine_reports_confidence=False,
            )
        )
        result = extract_text(_make_image(), engine=engine)
        assert len(result.metadata["regions"]) == 2
        assert result.metadata["regions"][0]["type"] == "header"
        assert result.metadata["regions"][0]["text"] == ""
        assert len(result.words) == 1
        assert result.words[0].text == "Real text"

    def test_metadata_omits_region_keys_when_engine_does_not_report_them(self):
        # An engine that only implements the base word-level contract
        # (no region_type) must not get fabricated regions/tables keys.
        engine = FakeOCREngine(_canned_raw_data([("X", 90.0, (0, 0, 10, 10))]))
        result = extract_text(_make_image(), engine=engine)
        assert "regions" not in result.metadata
        assert "tables" not in result.metadata

    def test_confidence_available_defaults_true_when_engine_silent_about_it(self):
        engine = FakeOCREngine(_canned_raw_data([("X", 90.0, (0, 0, 10, 10))]))
        result = extract_text(_make_image(), engine=engine)
        assert result.metadata["confidence_available"] is True


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


class TestErrors:
    def test_engine_failure_becomes_ocr_error(self):
        engine = FakeOCREngine(error=OCRError("simulated engine failure"))
        with pytest.raises(OCRError):
            extract_text(_make_image(), engine=engine)


# --------------------------------------------------------------------------
# extract_text_safe
# --------------------------------------------------------------------------


class TestExtractTextSafe:
    def test_returns_result_on_success(self):
        engine = FakeOCREngine(_canned_raw_data([("HELLO", 90.0, (0, 0, 10, 10))]))
        result = extract_text_safe(_make_image(), engine=engine)
        assert isinstance(result, PageOCRResult)
        assert "HELLO" in result.text

    def test_returns_none_instead_of_raising_on_invalid_image(self):
        assert extract_text_safe(None, engine=FakeOCREngine()) is None

    def test_returns_none_on_engine_failure(self):
        engine = FakeOCREngine(error=OCRError("simulated engine failure"))
        assert extract_text_safe(_make_image(), engine=engine) is None


# --------------------------------------------------------------------------
# Language configuration (English / Arabic / bilingual)
# --------------------------------------------------------------------------


class TestLanguageConfiguration:
    def test_default_language_is_bilingual(self):
        # Per project requirement: don't privilege English or Arabic.
        assert OCRConfig().language == "ara+eng"

    def test_language_is_carried_through_to_result(self):
        engine = FakeOCREngine(_canned_raw_data([]))
        result = extract_text(_make_image(), config=OCRConfig(language="eng"), engine=engine)
        assert result.language == "eng"

    def test_engine_receives_the_configured_language(self):
        engine = FakeOCREngine(_canned_raw_data([]))
        extract_text(_make_image(), config=OCRConfig(language="ara+eng"), engine=engine)
        assert engine.calls[-1][1] == "ara+eng"

    def test_bilingual_language_round_trips_through_result(self):
        engine = FakeOCREngine(_canned_raw_data([("HELLO", 90.0, (0, 0, 10, 10))]))
        result = extract_text(_make_image(), config=OCRConfig(language="ara+eng"), engine=engine)
        assert result.language == "ara+eng"
        assert "HELLO" in result.text


# --------------------------------------------------------------------------
# Confidence bucketing
# --------------------------------------------------------------------------


class TestConfidenceBucketing:
    def test_very_good_threshold(self):
        assert confidence_level(95.0) == ConfidenceLevel.VERY_GOOD

    def test_good_threshold(self):
        assert confidence_level(80.0) == ConfidenceLevel.GOOD

    def test_moderate_threshold(self):
        assert confidence_level(60.0) == ConfidenceLevel.MODERATE

    def test_poor_threshold(self):
        assert confidence_level(10.0) == ConfidenceLevel.POOR

    def test_boundaries_are_inclusive_of_the_higher_bucket(self):
        assert confidence_level(90.0) == ConfidenceLevel.VERY_GOOD
        assert confidence_level(75.0) == ConfidenceLevel.GOOD
        assert confidence_level(50.0) == ConfidenceLevel.MODERATE

    def test_custom_thresholds_are_respected(self):
        custom = ConfidenceThresholds(very_good=99.0, good=95.0, moderate=90.0)
        assert confidence_level(96.0, custom) == ConfidenceLevel.GOOD

    def test_result_carries_expected_confidence_level(self):
        engine = FakeOCREngine(_canned_raw_data([("HELLO", 90.0, (0, 0, 10, 10))]))
        result = extract_text(_make_image(), engine=engine)
        assert result.confidence_level in ConfidenceLevel


# --------------------------------------------------------------------------
# Engine injection / extensibility
# --------------------------------------------------------------------------


class TestEngineExtensibility:
    def test_extract_text_accepts_a_custom_engine(self):
        engine = FakeOCREngine(_canned_raw_data([("FAKE", 88.0, (0, 0, 40, 20))]))
        result = extract_text(_make_image(), engine=engine)
        assert result.text == "FAKE"
        assert result.engine == "fake"

    def test_custom_engine_bypasses_the_configured_default_engine(self, monkeypatch):
        import src.ocr.factory as ocr_factory

        def _fail_if_called(*args, **kwargs):
            raise AssertionError("The configured default engine should not have been built")

        monkeypatch.setattr(ocr_factory, "get_ocr_engine", _fail_if_called)
        engine = FakeOCREngine(_canned_raw_data([("OK", 99.0, (0, 0, 10, 10))]))
        result = extract_text(_make_image(), engine=engine)
        assert result.text == "OK"

    def test_engine_must_implement_recognize_raw(self):
        with pytest.raises(TypeError):
            OCREngine()  # abstract; cannot be instantiated directly


# --------------------------------------------------------------------------
# Preprocessing -> OCR integration
# --------------------------------------------------------------------------


class TestPreprocessingIntegration:
    def test_ocr_runs_on_preprocessed_output(self):
        from src.preprocessing.preprocessing import preprocess_page
        from src.quality.image_quality import assess_page_quality

        image = _make_image(width=1200, height=300)
        report = assess_page_quality(image, page_number=1)
        processed = preprocess_page(image, report)

        engine = FakeOCREngine(_canned_raw_data([("HELLO", 90.0, (0, 0, 10, 10))]))
        result = extract_text(processed.image, page_number=1, engine=engine)
        assert "HELLO" in result.text
        assert result.page_number == 1
        # Confirms the real preprocessed image (not the original) is
        # what actually reached the engine.
        assert engine.calls[-1][0] is processed.image


# --------------------------------------------------------------------------
# OCREngine.process() -- standardized entry point
# --------------------------------------------------------------------------


class TestEngineProcessMethod:
    def test_process_returns_page_ocr_result(self):
        engine = FakeOCREngine(_canned_raw_data([("HELLO", 90.0, (0, 0, 10, 10))]))
        result = engine.process(_make_image())
        assert isinstance(result, PageOCRResult)
        assert "HELLO" in result.text
        assert result.engine == "fake"

    def test_process_records_processing_time(self):
        engine = FakeOCREngine(_canned_raw_data([]))
        result = engine.process(_make_image())
        assert result.processing_time_ms is not None
        assert result.processing_time_ms >= 0

    def test_process_accepts_language_override(self):
        engine = FakeOCREngine(_canned_raw_data([("HELLO", 90.0, (0, 0, 10, 10))]))
        result = engine.process(_make_image(), language="eng")
        assert result.language == "eng"

    def test_base_class_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            OCREngine()


# --------------------------------------------------------------------------
# OCR engine factory (see also tests/test_unlimited_ocr.py for
# UnlimitedOCREngine-specific construction/parsing tests)
# --------------------------------------------------------------------------


class TestOCRFactory:
    def test_get_unlimited_engine_by_name(self):
        from src.ocr.factory import get_ocr_engine
        from src.ocr.unlimited_ocr import UnlimitedOCREngine

        engine = get_ocr_engine("unlimited")
        assert isinstance(engine, UnlimitedOCREngine)
        assert engine.name == "unlimited_ocr"

    def test_name_lookup_is_case_insensitive(self):
        from src.ocr.factory import get_ocr_engine

        assert get_ocr_engine("UNLIMITED").name == "unlimited_ocr"

    def test_defaults_to_config_engine_when_name_omitted(self):
        from src.ocr.factory import get_ocr_engine
        from src.ocr.unlimited_ocr import UnlimitedOCREngine

        engine = get_ocr_engine(config=OCRConfig(engine="unlimited"))
        assert isinstance(engine, UnlimitedOCREngine)

    def test_factory_construction_does_not_require_torch_or_transformers(self):
        # UnlimitedOCREngine loads its model lazily on first
        # recognize_raw() call, not at construction time -- so simply
        # building the engine (what get_ocr_engine does) must succeed
        # even on a machine without torch/transformers/a GPU installed.
        from src.ocr.factory import get_ocr_engine

        engine = get_ocr_engine("unlimited")
        assert engine is not None

    @pytest.mark.parametrize("engine_name", ["tesseract", "easyocr", "paddleocr"])
    def test_removed_or_never_implemented_engine_names_raise_controlled_error(self, engine_name):
        # Requirement: a non-"unlimited" engine explicitly requested
        # must fail clearly, not be silently ignored/substituted.
        from src.ocr.factory import get_ocr_engine
        from src.ocr.models import OCREngineNotAvailableError

        with pytest.raises(OCREngineNotAvailableError, match=engine_name):
            get_ocr_engine(engine_name)

    def test_unknown_engine_name_raises_controlled_error(self):
        from src.ocr.factory import get_ocr_engine
        from src.ocr.models import OCREngineNotAvailableError

        with pytest.raises(OCREngineNotAvailableError, match="unknown_engine_xyz"):
            get_ocr_engine("unknown_engine_xyz")


# --------------------------------------------------------------------------
# Engine selection via configuration (OCR_ENGINE)
# --------------------------------------------------------------------------


class TestOCREngineConfiguration:
    def test_config_default_engine_is_unlimited(self):
        assert OCRConfig().engine == "unlimited"

    def test_extract_text_uses_config_engine_when_none_given(self, monkeypatch):
        # No `engine=` kwarg passed -- extract_text must resolve the
        # engine via OCRConfig.engine / the factory, not a hardcoded
        # class. recognize_raw is monkeypatched so this never touches
        # torch/transformers/a real model.
        from src.ocr.unlimited_ocr import UnlimitedOCREngine

        def _fake_recognize_raw(self, image, language):
            return _canned_raw_data([("FAKE", 0.0, (0, 0, 10, 10))], engine_reports_confidence=False)

        monkeypatch.setattr(UnlimitedOCREngine, "recognize_raw", _fake_recognize_raw)

        result = extract_text(_make_image(), config=OCRConfig(engine="unlimited"))
        assert result.engine == "unlimited_ocr"
        assert result.text == "FAKE"

    def test_extract_text_with_unsupported_engine_raises(self):
        from src.ocr.models import OCREngineNotAvailableError

        with pytest.raises(OCREngineNotAvailableError):
            extract_text(_make_image(), config=OCRConfig(engine="paddleocr"))

    @pytest.mark.parametrize("engine_name", ["tesseract", "easyocr"])
    def test_extract_text_with_removed_engine_name_raises(self, engine_name):
        from src.ocr.models import OCREngineNotAvailableError

        with pytest.raises(OCREngineNotAvailableError):
            extract_text(_make_image(), config=OCRConfig(engine=engine_name))
