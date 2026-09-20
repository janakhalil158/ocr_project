"""
Tests for Phase 4: Tesseract OCR.

Follows the same approach as ``test_preprocessing.py``: all test images
are generated synthetically with NumPy/OpenCV rather than loaded from
real-world files. Unlike Phase 2/3's tests, these images render actual
text (via ``cv2.putText``) since OCR has to have real characters to
recognize. Text assertions allow reasonable OCR variation (checking
that expected words appear, rather than requiring an exact string
match) since Tesseract's precise output can shift slightly across
versions/platforms.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import pytesseract

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


def _make_single_line_image(text: str = "HELLO WORLD", width: int = 700, height: int = 150) -> np.ndarray:
    image = np.full((height, width), 255, dtype=np.uint8)
    cv2.putText(image, text, (20, height // 2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 1.4, 0, 2, cv2.LINE_AA)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)


def _make_multi_line_image(
    lines=("HELLO WORLD", "SECOND LINE"), width: int = 700, line_height: int = 100
) -> np.ndarray:
    height = line_height * len(lines) + 40
    image = np.full((height, width), 255, dtype=np.uint8)
    y = 70
    for line in lines:
        cv2.putText(image, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.3, 0, 2, cv2.LINE_AA)
        y += line_height
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)


def _make_blank_image(width: int = 500, height: int = 300) -> np.ndarray:
    return np.full((height, width, 3), 255, dtype=np.uint8)


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------


class TestInputValidation:
    def test_none_image_raises(self):
        with pytest.raises(OCRError):
            extract_text(None)

    def test_wrong_type_raises(self):
        with pytest.raises(OCRError):
            extract_text([[1, 2], [3, 4]])

    def test_wrong_dtype_raises(self):
        image = np.full((100, 100, 3), 255, dtype=np.float32)
        with pytest.raises(OCRError):
            extract_text(image)

    def test_too_small_image_raises(self):
        image = np.full((2, 2, 3), 255, dtype=np.uint8)
        with pytest.raises(OCRError):
            extract_text(image)

    def test_rgb_input_supported(self):
        image = _make_single_line_image()
        result = extract_text(image)
        assert "HELLO" in result.text

    def test_grayscale_input_supported(self):
        image = cv2.cvtColor(_make_single_line_image(), cv2.COLOR_RGB2GRAY)
        result = extract_text(image)
        assert "HELLO" in result.text


# --------------------------------------------------------------------------
# Text recognition
# --------------------------------------------------------------------------


class TestTextRecognition:
    def test_known_text_recognized(self):
        image = _make_single_line_image("HELLO WORLD")
        result = extract_text(image)
        assert "HELLO" in result.text
        assert "WORLD" in result.text

    def test_multiple_words_produce_multiple_word_results(self):
        image = _make_single_line_image("HELLO WORLD")
        result = extract_text(image)
        assert len(result.words) == 2

    def test_multiple_lines_preserved_in_text(self):
        image = _make_multi_line_image(("HELLO WORLD", "SECOND LINE"))
        result = extract_text(image)
        lines = result.text.split("\n")
        assert len(lines) == 2
        assert "HELLO" in lines[0] and "WORLD" in lines[0]
        assert "SECOND" in lines[1] and "LINE" in lines[1]

    def test_blank_page_returns_empty_text(self):
        image = _make_blank_image()
        result = extract_text(image)
        assert result.text == ""


# --------------------------------------------------------------------------
# Result structure
# --------------------------------------------------------------------------


class TestResultStructure:
    def test_result_is_page_ocr_result_instance(self):
        result = extract_text(_make_single_line_image())
        assert isinstance(result, PageOCRResult)

    def test_words_are_word_result_instances(self):
        result = extract_text(_make_single_line_image())
        assert all(isinstance(w, WordResult) for w in result.words)

    def test_bounding_boxes_are_bounding_box_instances(self):
        result = extract_text(_make_single_line_image())
        assert all(isinstance(w.bounding_box, BoundingBox) for w in result.words)

    def test_page_number_carried_through(self):
        result = extract_text(_make_single_line_image(), page_number=42)
        assert result.page_number == 42

    def test_to_dict_matches_expected_shape(self):
        result = extract_text(_make_single_line_image())
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
        result = extract_text(_make_single_line_image())
        assert result.words  # sanity: the image should actually produce words
        for word in result.words:
            assert 0.0 <= word.confidence <= 100.0

    def test_mean_confidence_matches_manual_average(self):
        result = extract_text(_make_single_line_image("HELLO WORLD"))
        expected = sum(w.confidence for w in result.words) / len(result.words)
        assert result.mean_confidence == pytest.approx(expected)

    def test_blank_page_has_zero_mean_confidence_and_no_words(self):
        result = extract_text(_make_blank_image())
        assert result.words == []
        assert result.mean_confidence == 0.0

    def test_min_confidence_filters_out_words(self):
        # No word can score above 100, so a threshold of 101 must
        # exclude every word, leaving an empty result.
        image = _make_single_line_image("HELLO WORLD")
        result = extract_text(image, config=OCRConfig(min_confidence=101))
        assert result.words == []
        assert result.text == ""
        assert result.mean_confidence == 0.0


# --------------------------------------------------------------------------
# Bounding boxes
# --------------------------------------------------------------------------


class TestBoundingBoxes:
    def test_boxes_exist_for_every_word(self):
        result = extract_text(_make_single_line_image("HELLO WORLD"))
        assert len(result.words) == 2
        assert all(w.bounding_box is not None for w in result.words)

    def test_coordinates_are_non_negative(self):
        result = extract_text(_make_single_line_image())
        for word in result.words:
            box = word.bounding_box
            assert box.x >= 0 and box.y >= 0

    def test_width_and_height_are_positive(self):
        result = extract_text(_make_single_line_image())
        for word in result.words:
            box = word.bounding_box
            assert box.width > 0 and box.height > 0

    def test_second_word_positioned_right_of_first(self):
        # "HELLO WORLD" on one line: WORLD should sit to the right of HELLO.
        result = extract_text(_make_single_line_image("HELLO WORLD"))
        hello, world = result.words[0], result.words[1]
        assert world.bounding_box.x > hello.bounding_box.x


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


class TestErrors:
    def test_tesseract_not_found_becomes_ocr_error(self, monkeypatch):
        import pytesseract

        def _raise_not_found(*args, **kwargs):
            raise pytesseract.TesseractNotFoundError()

        monkeypatch.setattr(pytesseract, "image_to_data", _raise_not_found)
        with pytest.raises(OCRError):
            extract_text(_make_single_line_image())

    def test_tesseract_execution_failure_becomes_ocr_error(self, monkeypatch):
        import pytesseract

        def _raise_tesseract_error(*args, **kwargs):
            raise pytesseract.TesseractError(1, "simulated failure")

        monkeypatch.setattr(pytesseract, "image_to_data", _raise_tesseract_error)
        with pytest.raises(OCRError):
            extract_text(_make_single_line_image())


# --------------------------------------------------------------------------
# extract_text_safe
# --------------------------------------------------------------------------


class TestExtractTextSafe:
    def test_returns_result_on_success(self):
        result = extract_text_safe(_make_single_line_image("HELLO WORLD"))
        assert isinstance(result, PageOCRResult)
        assert "HELLO" in result.text

    def test_returns_none_instead_of_raising_on_invalid_image(self):
        assert extract_text_safe(None) is None

    def test_returns_none_on_engine_failure(self, monkeypatch):
        import pytesseract

        def _raise_not_found(*args, **kwargs):
            raise pytesseract.TesseractNotFoundError()

        monkeypatch.setattr(pytesseract, "image_to_data", _raise_not_found)
        assert extract_text_safe(_make_single_line_image()) is None


# --------------------------------------------------------------------------
# Language configuration (English / Arabic / bilingual)
# --------------------------------------------------------------------------


class TestLanguageConfiguration:
    def test_default_language_is_bilingual(self):
        # Per project requirement: don't privilege English or Arabic.
        assert OCRConfig().language == "ara+eng"

    def test_english_only_language_recognizes_english(self):
        image = _make_single_line_image("HELLO WORLD")
        result = extract_text(image, config=OCRConfig(language="eng"))
        assert result.language == "eng"
        assert "HELLO" in result.text

    def test_bilingual_language_still_recognizes_english(self):
        image = _make_single_line_image("HELLO WORLD")
        result = extract_text(image, config=OCRConfig(language="ara+eng"))
        assert result.language == "ara+eng"
        assert "HELLO" in result.text

    def test_language_is_carried_through_to_result(self):
        result = extract_text(_make_single_line_image(), config=OCRConfig(language="eng"))
        assert result.language == "eng"


def _make_arabic_line_image(text: str = "مرحبا بالعالم", width: int = 700, height: int = 160) -> np.ndarray:
    """
    Render a line of Arabic text using PIL + an Arabic-capable font.

    ``cv2.putText`` cannot render Arabic glyphs at all, so this helper
    uses PIL with the KACST font instead, with proper shaping/bidi via
    ``arabic_reshaper``/``python-bidi`` so joined letterforms come out
    the way Tesseract's Arabic model expects.
    """
    from PIL import Image, ImageDraw, ImageFont

    reshaped = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped)

    image = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(_ARABIC_FONT_PATH, 48)
    draw.text((20, height // 2 - 30), bidi_text, font=font, fill=(0, 0, 0))
    return np.array(image)


try:
    import arabic_reshaper
    from bidi.algorithm import get_display

    _ARABIC_FONT_PATH = "/usr/share/fonts/truetype/kacst/KacstBook.ttf"
    import os as _os

    _ARABIC_DEPS_AVAILABLE = _os.path.exists(_ARABIC_FONT_PATH)
except ImportError:
    _ARABIC_DEPS_AVAILABLE = False

_skip_without_arabic_deps = pytest.mark.skipif(
    not _ARABIC_DEPS_AVAILABLE,
    reason=(
        "Arabic test-image generation requires arabic_reshaper, python-bidi, "
        "and an Arabic-capable font (e.g. fonts-kacst) to be installed; these "
        "are only needed to build synthetic test images, not by the OCR "
        "pipeline itself."
    ),
)

_skip_without_arabic_traineddata = pytest.mark.skipif(
    "ara" not in pytesseract.get_languages(config=""),
    reason="Tesseract's 'ara' language pack is not installed on this machine.",
)


class TestArabicOCR:
    @_skip_without_arabic_deps
    @_skip_without_arabic_traineddata
    def test_arabic_text_produces_nonempty_result(self):
        image = _make_arabic_line_image()
        result = extract_text(image, config=OCRConfig(language="ara"))
        assert result.text_detected
        assert len(result.words) > 0

    @_skip_without_arabic_deps
    @_skip_without_arabic_traineddata
    def test_arabic_with_bilingual_config_still_produces_result(self):
        image = _make_arabic_line_image()
        result = extract_text(image, config=OCRConfig(language="ara+eng"))
        assert result.text_detected


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
        result = extract_text(_make_single_line_image("HELLO WORLD"))
        assert result.confidence_level in ConfidenceLevel


# --------------------------------------------------------------------------
# Engine injection / extensibility
# --------------------------------------------------------------------------


class _StubEngine(OCREngine):
    """Minimal fake OCREngine to prove the orchestration layer is engine-agnostic."""

    name = "stub"

    def __init__(self, canned_response):
        self._canned_response = canned_response

    def recognize_raw(self, image, language):
        return self._canned_response


def _canned_tesseract_shape(words_with_boxes):
    """Build a minimal Tesseract-shaped raw dict for the given words."""
    data = {
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
    for i, (text, conf, box) in enumerate(words_with_boxes):
        data["text"].append(text)
        data["conf"].append(conf)
        data["left"].append(box[0])
        data["top"].append(box[1])
        data["width"].append(box[2])
        data["height"].append(box[3])
        data["block_num"].append(1)
        data["par_num"].append(1)
        data["line_num"].append(1)
    return data


class TestEngineExtensibility:
    def test_extract_text_accepts_a_custom_engine(self):
        canned = _canned_tesseract_shape([("FAKE", 88.0, (0, 0, 40, 20))])
        engine = _StubEngine(canned)
        result = extract_text(_make_single_line_image(), engine=engine)
        assert result.text == "FAKE"
        assert result.engine == "stub"

    def test_custom_engine_bypasses_tesseract_entirely(self, monkeypatch):
        import pytesseract

        def _fail_if_called(*args, **kwargs):
            raise AssertionError("Tesseract should not have been called")

        monkeypatch.setattr(pytesseract, "image_to_data", _fail_if_called)
        canned = _canned_tesseract_shape([("OK", 99.0, (0, 0, 10, 10))])
        result = extract_text(_make_single_line_image(), engine=_StubEngine(canned))
        assert result.text == "OK"

    def test_engine_must_implement_recognize_raw(self):
        with pytest.raises(TypeError):
            OCREngine()  # abstract; cannot be instantiated directly


# --------------------------------------------------------------------------
# Preprocessing -> OCR integration
# --------------------------------------------------------------------------


class TestPreprocessingIntegration:
    def test_ocr_runs_on_preprocessed_output(self):
        from src.quality.image_quality import assess_page_quality
        from src.preprocessing.preprocessing import preprocess_page

        image = _make_single_line_image("HELLO WORLD", width=1200, height=300)
        report = assess_page_quality(image, page_number=1)
        processed = preprocess_page(image, report)

        result = extract_text(processed.image, page_number=1)
        assert "HELLO" in result.text
        assert result.page_number == 1


# --------------------------------------------------------------------------
# OCREngine.process() -- standardized entry point
# --------------------------------------------------------------------------


class TestEngineProcessMethod:
    def test_process_returns_page_ocr_result(self):
        from src.ocr.tesseract_ocr import TesseractOCREngine

        engine = TesseractOCREngine()
        result = engine.process(_make_single_line_image("HELLO WORLD"))
        assert isinstance(result, PageOCRResult)
        assert "HELLO" in result.text
        assert result.engine == "tesseract"

    def test_process_records_processing_time(self):
        from src.ocr.tesseract_ocr import TesseractOCREngine

        result = TesseractOCREngine().process(_make_single_line_image())
        assert result.processing_time_ms is not None
        assert result.processing_time_ms >= 0

    def test_process_accepts_language_override(self):
        from src.ocr.tesseract_ocr import TesseractOCREngine

        result = TesseractOCREngine().process(_make_single_line_image("HELLO WORLD"), language="eng")
        assert result.language == "eng"

    def test_process_works_through_a_custom_engine(self):
        canned = _canned_tesseract_shape([("STUB", 77.0, (0, 0, 30, 15))])
        engine = _StubEngine(canned)
        result = engine.process(_make_single_line_image())
        assert result.text == "STUB"
        assert result.engine == "stub"

    def test_base_class_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            OCREngine()


# --------------------------------------------------------------------------
# OCR engine factory
# --------------------------------------------------------------------------


class TestOCRFactory:
    def test_get_tesseract_engine_by_name(self):
        from src.ocr.factory import get_ocr_engine
        from src.ocr.tesseract_ocr import TesseractOCREngine

        engine = get_ocr_engine("tesseract")
        assert isinstance(engine, TesseractOCREngine)
        assert engine.name == "tesseract"

    def test_name_lookup_is_case_insensitive(self):
        from src.ocr.factory import get_ocr_engine

        assert get_ocr_engine("TESSERACT").name == "tesseract"

    def test_defaults_to_config_engine_when_name_omitted(self):
        from src.ocr.factory import get_ocr_engine
        from src.ocr.tesseract_ocr import TesseractOCREngine

        engine = get_ocr_engine(config=OCRConfig(engine="tesseract"))
        assert isinstance(engine, TesseractOCREngine)

    def test_paddleocr_raises_controlled_informative_error(self):
        from src.ocr.factory import get_ocr_engine
        from src.ocr.models import OCREngineNotAvailableError

        with pytest.raises(OCREngineNotAvailableError, match="paddleocr"):
            get_ocr_engine("paddleocr")

    def test_paddleocr_does_not_attempt_an_import(self):
        # A controlled error, not a bare ImportError surfacing from deep
        # inside the pipeline -- this must succeed whether or not the
        # paddleocr package happens to be installed on this machine.
        import sys

        from src.ocr.factory import get_ocr_engine
        from src.ocr.models import OCREngineNotAvailableError

        assert "paddleocr" not in sys.modules
        with pytest.raises(OCREngineNotAvailableError):
            get_ocr_engine("paddleocr")
        assert "paddleocr" not in sys.modules

    def test_unknown_engine_name_raises_controlled_error(self):
        from src.ocr.factory import get_ocr_engine
        from src.ocr.models import OCREngineNotAvailableError

        with pytest.raises(OCREngineNotAvailableError, match="unknown_engine_xyz"):
            get_ocr_engine("unknown_engine_xyz")


# --------------------------------------------------------------------------
# Engine selection via configuration (OCR_ENGINE)
# --------------------------------------------------------------------------


class TestOCREngineConfiguration:
    def test_config_default_engine_is_tesseract(self):
        assert OCRConfig().engine == "tesseract"

    def test_extract_text_uses_config_engine_when_none_given(self):
        # No `engine=` kwarg passed -- extract_text must resolve the
        # engine via OCRConfig.engine / the factory, not a hardcoded class.
        result = extract_text(_make_single_line_image("HELLO WORLD"), config=OCRConfig(engine="tesseract"))
        assert result.engine == "tesseract"

    def test_extract_text_with_unavailable_configured_engine_raises(self):
        from src.ocr.models import OCREngineNotAvailableError

        with pytest.raises(OCREngineNotAvailableError):
            extract_text(_make_single_line_image(), config=OCRConfig(engine="paddleocr"))
