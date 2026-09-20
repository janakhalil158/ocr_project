"""
Tests for Phase 4: EasyOCR engine (src/ocr/easyocr_ocr.py).

EasyOCR's real package requires downloading detection/recognition
models on first use, which would make the test suite slow and
network-dependent. Every test here injects a small fake ``easyocr``
module (``_FakeEasyOCRModule`` / ``_FakeReader`` below) via
``EasyOCREngine(..., easyocr_module=...)`` instead, so these tests never
need the real package installed and never touch the network — matching
the same synthetic-data approach already used by test_ocr.py /
test_preprocessing.py for the rest of this pipeline.
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import pytest

from src.ocr.easyocr_ocr import (
    EasyOCREngine,
    _bbox_to_ltwh,
    _import_easyocr,
    convert_language,
)
from src.ocr.factory import SUPPORTED_ENGINES, get_ocr_engine
from src.ocr.models import OCREngineNotAvailableError, OCRError
from src.ocr.ocr import extract_text
from src.utils.config import OCRConfig


# --------------------------------------------------------------------------
# Fake `easyocr` module: no real install, no model downloads, deterministic.
# --------------------------------------------------------------------------


class _FakeReader:
    def __init__(self, languages, gpu=False, readtext_result=None, raise_on_readtext=None):
        self.languages = languages
        self.gpu = gpu
        self._readtext_result = readtext_result if readtext_result is not None else []
        self._raise_on_readtext = raise_on_readtext
        self.readtext_call_count = 0

    def readtext(self, image, detail=1):
        self.readtext_call_count += 1
        if self._raise_on_readtext is not None:
            raise self._raise_on_readtext
        return self._readtext_result


class _FakeEasyOCRModule:
    """Stand-in for the real `easyocr` package."""

    def __init__(self, readtext_result=None, raise_on_readtext=None, raise_on_reader_init=None):
        self.reader_init_calls = []
        self._readtext_result = readtext_result
        self._raise_on_readtext = raise_on_readtext
        self._raise_on_reader_init = raise_on_reader_init

    def Reader(self, languages, gpu=False):
        self.reader_init_calls.append({"languages": list(languages), "gpu": gpu})
        if self._raise_on_reader_init is not None:
            raise self._raise_on_reader_init
        return _FakeReader(
            languages,
            gpu,
            readtext_result=self._readtext_result,
            raise_on_readtext=self._raise_on_readtext,
        )


def _make_image(width: int = 100, height: int = 50) -> np.ndarray:
    return np.full((height, width, 3), 255, dtype=np.uint8)


# --------------------------------------------------------------------------
# convert_language
# --------------------------------------------------------------------------


class TestConvertLanguage:
    def test_project_default_ara_eng_maps_to_ar_en(self) -> None:
        assert convert_language("ara+eng") == ["ar", "en"]

    def test_single_language(self) -> None:
        assert convert_language("eng") == ["en"]
        assert convert_language("ara") == ["ar"]

    def test_empty_or_none_falls_back_to_default_languages(self) -> None:
        assert convert_language("") == list(EasyOCREngine.DEFAULT_LANGUAGES)
        assert convert_language(None) == list(EasyOCREngine.DEFAULT_LANGUAGES)

    def test_unmapped_token_passes_through_lowercased_rather_than_breaking(self) -> None:
        # 'fra' has no explicit mapping -- must not raise or silently drop it.
        assert convert_language("fra+eng") == ["fra", "en"]

    def test_duplicate_tokens_are_not_repeated(self) -> None:
        assert convert_language("eng+eng") == ["en"]


# --------------------------------------------------------------------------
# _bbox_to_ltwh
# --------------------------------------------------------------------------


class TestBoundingBoxConversion:
    def test_axis_aligned_box(self) -> None:
        points = [[10, 20], [50, 20], [50, 40], [10, 40]]
        left, top, width, height = _bbox_to_ltwh(points)
        assert (left, top, width, height) == (10.0, 20.0, 40.0, 20.0)

    def test_non_axis_aligned_box_uses_min_max_extent(self) -> None:
        # A slightly rotated quadrilateral -- left/top/width/height should
        # be the axis-aligned bounding box of all four points.
        points = [[12, 18], [52, 22], [48, 42], [8, 38]]
        left, top, width, height = _bbox_to_ltwh(points)
        assert left == 8.0
        assert top == 18.0
        assert width == 52 - 8
        assert height == 42 - 18


# --------------------------------------------------------------------------
# _import_easyocr
# --------------------------------------------------------------------------


class TestImportEasyOCR:
    def test_missing_package_raises_controlled_error(self, monkeypatch) -> None:
        # Force `import easyocr` to fail deterministically regardless of
        # whether the real package happens to be installed in this
        # environment (setting sys.modules[name] = None makes Python's
        # import system raise ImportError for that name).
        monkeypatch.setitem(sys.modules, "easyocr", None)

        with pytest.raises(OCREngineNotAvailableError):
            _import_easyocr()


# --------------------------------------------------------------------------
# EasyOCREngine construction
# --------------------------------------------------------------------------


class TestEasyOCREngineConstruction:
    def test_default_languages(self) -> None:
        fake_module = _FakeEasyOCRModule()
        engine = EasyOCREngine(easyocr_module=fake_module)
        assert engine.languages == list(EasyOCREngine.DEFAULT_LANGUAGES)

    def test_construction_does_not_create_reader(self) -> None:
        """
        Constructing an EasyOCREngine must NOT build the (expensive,
        model-loading) Reader. It's created lazily, on first use --
        see TestReaderLazyInitAndReuse below.
        """
        fake_module = _FakeEasyOCRModule()
        EasyOCREngine(languages=["ar", "en"], easyocr_module=fake_module)
        assert fake_module.reader_init_calls == []

    def test_explicit_languages_passed_through_to_reader(self) -> None:
        fake_module = _FakeEasyOCRModule()
        engine = EasyOCREngine(languages=["ar", "en"], easyocr_module=fake_module)
        engine.recognize_raw(_make_image(), language="ara+eng")
        assert fake_module.reader_init_calls == [{"languages": ["ar", "en"], "gpu": False}]

    def test_defaults_to_cpu_mode(self) -> None:
        fake_module = _FakeEasyOCRModule()
        engine = EasyOCREngine(easyocr_module=fake_module)
        assert engine.gpu is False
        engine.recognize_raw(_make_image(), language="eng")
        assert fake_module.reader_init_calls[0]["gpu"] is False

    def test_gpu_flag_is_honored_when_explicitly_requested(self) -> None:
        fake_module = _FakeEasyOCRModule()
        engine = EasyOCREngine(gpu=True, easyocr_module=fake_module)
        assert engine.gpu is True
        engine.recognize_raw(_make_image(), language="eng")
        assert fake_module.reader_init_calls[0]["gpu"] is True

    def test_engine_name_is_easyocr(self) -> None:
        fake_module = _FakeEasyOCRModule()
        engine = EasyOCREngine(easyocr_module=fake_module)
        assert engine.name == "easyocr"

    def test_reader_init_failure_raises_controlled_error(self) -> None:
        """
        Since the Reader is now built lazily, a construction-time
        failure surfaces on the first recognize_raw() call, not at
        EasyOCREngine(...) construction time itself.
        """
        fake_module = _FakeEasyOCRModule(raise_on_reader_init=RuntimeError("boom"))
        engine = EasyOCREngine(easyocr_module=fake_module)
        with pytest.raises(OCREngineNotAvailableError):
            engine.recognize_raw(_make_image(), language="eng")


# --------------------------------------------------------------------------
# recognize_raw
# --------------------------------------------------------------------------


class TestRecognizeRaw:
    def test_normalized_dict_has_expected_keys_and_equal_length_lists(self) -> None:
        fake_module = _FakeEasyOCRModule(
            readtext_result=[
                ([[10, 20], [50, 20], [50, 40], [10, 40]], "HELLO", 0.87),
                ([[10, 60], [90, 60], [90, 80], [10, 80]], "WORLD", 0.5),
            ]
        )
        engine = EasyOCREngine(easyocr_module=fake_module)
        data = engine.recognize_raw(_make_image(), language="ara+eng")

        expected_keys = {
            "text", "conf", "left", "top", "width", "height",
            "block_num", "par_num", "line_num",
        }
        assert set(data.keys()) == expected_keys

        lengths = {len(v) for v in data.values()}
        assert lengths == {2}

    def test_bounding_boxes_are_converted_correctly(self) -> None:
        fake_module = _FakeEasyOCRModule(
            readtext_result=[([[10, 20], [50, 20], [50, 40], [10, 40]], "HELLO", 0.9)]
        )
        engine = EasyOCREngine(easyocr_module=fake_module)
        data = engine.recognize_raw(_make_image(), language="eng")

        assert data["left"] == [10]
        assert data["top"] == [20]
        assert data["width"] == [40]
        assert data["height"] == [20]

    def test_confidence_normalized_from_0_1_to_0_100(self) -> None:
        fake_module = _FakeEasyOCRModule(
            readtext_result=[
                ([[0, 0], [10, 0], [10, 10], [0, 10]], "A", 0.995),
                ([[0, 0], [10, 0], [10, 10], [0, 10]], "B", 0.0),
            ]
        )
        engine = EasyOCREngine(easyocr_module=fake_module)
        data = engine.recognize_raw(_make_image(), language="eng")
        assert data["conf"] == pytest.approx([99.5, 0.0])

    def test_text_and_confidence_alignment(self) -> None:
        fake_module = _FakeEasyOCRModule(
            readtext_result=[([[0, 0], [10, 0], [10, 10], [0, 10]], "HELLO WORLD", 0.7)]
        )
        engine = EasyOCREngine(easyocr_module=fake_module)
        data = engine.recognize_raw(_make_image(), language="eng")
        assert data["text"] == ["HELLO WORLD"]

    def test_each_detection_gets_its_own_line_number(self) -> None:
        fake_module = _FakeEasyOCRModule(
            readtext_result=[
                ([[0, 0], [10, 0], [10, 10], [0, 10]], "A", 0.9),
                ([[0, 20], [10, 20], [10, 30], [0, 30]], "B", 0.9),
            ]
        )
        engine = EasyOCREngine(easyocr_module=fake_module)
        data = engine.recognize_raw(_make_image(), language="eng")
        assert data["line_num"] == [1, 2]
        assert data["block_num"] == [1, 1]
        assert data["par_num"] == [1, 1]

    def test_empty_results_return_valid_empty_lists_without_crashing(self) -> None:
        fake_module = _FakeEasyOCRModule(readtext_result=[])
        engine = EasyOCREngine(easyocr_module=fake_module)
        data = engine.recognize_raw(_make_image(), language="eng")

        for key in ("text", "conf", "left", "top", "width", "height", "block_num", "par_num", "line_num"):
            assert data[key] == []

    def test_reader_failure_raises_ocr_error(self) -> None:
        fake_module = _FakeEasyOCRModule(raise_on_readtext=RuntimeError("inference failed"))
        engine = EasyOCREngine(easyocr_module=fake_module)
        with pytest.raises(OCRError):
            engine.recognize_raw(_make_image(), language="eng")

    def test_mismatched_language_does_not_rebuild_reader(self, caplog) -> None:
        """
        Passing a different `language` than the reader was built for must
        NOT reinitialize the (expensive, model-loading) reader -- it
        should log a warning and proceed with the existing reader.
        """
        fake_module = _FakeEasyOCRModule(readtext_result=[])
        engine = EasyOCREngine(languages=["en"], easyocr_module=fake_module)

        # First call, with the engine's own configured language, is what
        # lazily builds the reader.
        engine.recognize_raw(_make_image(), language="eng")

        assert len(fake_module.reader_init_calls) == 1

        # Second call, with a different language, must warn and reuse the
        # existing reader rather than rebuilding it.
        with caplog.at_level(logging.WARNING):
            engine.recognize_raw(_make_image(), language="ara+eng")

        assert any(
            record.levelno == logging.WARNING for record in caplog.records
        )
        assert len(fake_module.reader_init_calls) == 1  # unchanged


# --------------------------------------------------------------------------
# Full engine.process() path -> existing PageOCRResult
# --------------------------------------------------------------------------


class TestEasyOCREngineProcessIntegration:
    def test_process_produces_existing_page_ocr_result_shape(self) -> None:
        fake_module = _FakeEasyOCRModule(
            readtext_result=[([[10, 20], [110, 20], [110, 40], [10, 40]], "HELLO WORLD", 0.9)]
        )
        engine = EasyOCREngine(easyocr_module=fake_module)
        config = OCRConfig(language="eng", min_confidence=0.0)

        result = extract_text(_make_image(), config=config, engine=engine)

        assert result.engine == "easyocr"
        assert "HELLO WORLD" in result.text
        assert len(result.words) == 1
        assert result.words[0].confidence == pytest.approx(90.0)
        assert result.mean_confidence == pytest.approx(90.0)


# --------------------------------------------------------------------------
# Factory integration
# --------------------------------------------------------------------------


class TestFactoryIntegration:
    def test_easyocr_is_a_supported_engine(self) -> None:
        assert "easyocr" in SUPPORTED_ENGINES

    def test_get_ocr_engine_returns_easyocr_engine(self, monkeypatch) -> None:
        fake_module = _FakeEasyOCRModule()
        monkeypatch.setattr(
            "src.ocr.easyocr_ocr._import_easyocr", lambda: fake_module
        )

        engine = get_ocr_engine("easyocr")

        assert isinstance(engine, EasyOCREngine)
        assert engine.name == "easyocr"

    def test_get_ocr_engine_uses_config_language_and_gpu(self, monkeypatch) -> None:
        fake_module = _FakeEasyOCRModule()
        monkeypatch.setattr(
            "src.ocr.easyocr_ocr._import_easyocr", lambda: fake_module
        )

        config = OCRConfig(language="ara+eng", easyocr_gpu=False)
        engine = get_ocr_engine("easyocr", config=config)

        assert isinstance(engine, EasyOCREngine)
        # Initialization is lazy -- get_ocr_engine() alone must not have
        # built the reader yet.
        assert fake_module.reader_init_calls == []

        # The reader is only built on the first actual OCR call.
        engine.recognize_raw(_make_image(), language="ara+eng")

        assert fake_module.reader_init_calls == [{"languages": ["ar", "en"], "gpu": False}]