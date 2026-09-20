"""
Phase 5 tests: OCR evaluation & benchmarking.

The evaluator tests use a fake in-memory OCR engine rather than
Tesseract, so they verify the evaluation logic itself and run
identically on a machine with no Tesseract installed. The metric tests
are pure string math and need no engine at all.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.ocr.base import OCREngine
from src.ocr.evaluation import (
    EvaluationError,
    EvaluationResult,
    EvaluationSample,
    evaluate_dataset,
    evaluate_sample,
    load_dataset,
    save_results,
    summarize,
)
from src.ocr.metrics import (
    character_error_rate,
    levenshtein_distance,
    normalize_text,
    tokenize_words,
    word_error_rate,
)
from src.ocr.models import OCRError

ARABIC_LINE = "شهادة إتمام الدورة"


# --------------------------------------------------------------------------
# Levenshtein distance
# --------------------------------------------------------------------------


class TestLevenshteinDistance:
    def test_identical_sequences(self):
        assert levenshtein_distance("hello", "hello") == 0

    def test_single_substitution(self):
        assert levenshtein_distance("hello", "hallo") == 1

    def test_single_insertion(self):
        assert levenshtein_distance("hello", "helloo") == 1

    def test_single_deletion(self):
        assert levenshtein_distance("hello", "hell") == 1

    def test_empty_reference(self):
        assert levenshtein_distance("", "abc") == 3

    def test_empty_hypothesis(self):
        assert levenshtein_distance("abc", "") == 3

    def test_both_empty(self):
        assert levenshtein_distance("", "") == 0

    def test_works_on_word_lists(self):
        assert levenshtein_distance(["a", "b", "c"], ["a", "x", "c"]) == 1

    def test_is_symmetric(self):
        assert levenshtein_distance("kitten", "sitting") == levenshtein_distance("sitting", "kitten")


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------


class TestNormalization:
    def test_collapses_repeated_whitespace(self):
        assert normalize_text("a    b") == "a b"

    def test_strips_leading_and_trailing_whitespace(self):
        assert normalize_text("  hello  ") == "hello"

    def test_collapses_line_breaks_to_spaces(self):
        assert normalize_text("line one\nline two") == "line one line two"

    def test_handles_windows_line_endings(self):
        assert normalize_text("a\r\nb") == "a b"

    def test_empty_string_is_safe(self):
        assert normalize_text("") == ""

    def test_does_not_lowercase(self):
        # Case errors are real OCR errors and must not be normalized away.
        assert normalize_text("Hello") == "Hello"

    def test_does_not_strip_punctuation(self):
        assert normalize_text("hello, world!") == "hello, world!"

    def test_preserves_arabic_characters(self):
        assert normalize_text(ARABIC_LINE) == ARABIC_LINE

    def test_preserves_arabic_diacritics(self):
        # Tashkeel removal would destroy real character distinctions.
        with_diacritics = "مُحَمَّد"
        assert normalize_text(with_diacritics) == with_diacritics

    def test_tokenize_words_splits_on_whitespace(self):
        assert tokenize_words("one  two\nthree") == ["one", "two", "three"]

    def test_tokenize_empty_returns_empty_list(self):
        assert tokenize_words("") == []


# --------------------------------------------------------------------------
# Character Error Rate
# --------------------------------------------------------------------------


class TestCharacterErrorRate:
    def test_identical_strings_score_zero(self):
        assert character_error_rate("hello", "hello") == 0.0

    def test_single_substitution(self):
        assert character_error_rate("hello", "hallo") == pytest.approx(1 / 5)

    def test_single_insertion(self):
        assert character_error_rate("hello", "helloo") == pytest.approx(1 / 5)

    def test_single_deletion(self):
        assert character_error_rate("hello", "hell") == pytest.approx(1 / 5)

    def test_empty_hypothesis_scores_one(self):
        assert character_error_rate("hello", "") == pytest.approx(1.0)

    def test_empty_reference_and_empty_hypothesis_scores_zero(self):
        assert character_error_rate("", "") == 0.0

    def test_empty_reference_with_output_scores_one(self):
        # Documented convention, and no ZeroDivisionError.
        assert character_error_rate("", "spurious") == 1.0

    def test_whitespace_only_difference_scores_zero(self):
        assert character_error_rate("hello world", "hello    world") == 0.0

    def test_line_break_difference_scores_zero(self):
        assert character_error_rate("hello\nworld", "hello world") == 0.0

    def test_identical_arabic_scores_zero(self):
        assert character_error_rate(ARABIC_LINE, ARABIC_LINE) == 0.0

    def test_arabic_substitution_is_detected(self):
        assert character_error_rate(ARABIC_LINE, ARABIC_LINE[:-1] + "x") > 0.0

    def test_can_exceed_one_when_engine_hallucinates(self):
        assert character_error_rate("hi", "hi there friend") > 1.0


# --------------------------------------------------------------------------
# Word Error Rate
# --------------------------------------------------------------------------


class TestWordErrorRate:
    def test_identical_text_scores_zero(self):
        assert word_error_rate("the quick brown fox", "the quick brown fox") == 0.0

    def test_single_word_substitution(self):
        assert word_error_rate("the quick brown fox", "the slow brown fox") == pytest.approx(1 / 4)

    def test_single_word_insertion(self):
        assert word_error_rate("the quick fox", "the quick brown fox") == pytest.approx(1 / 3)

    def test_single_word_deletion(self):
        assert word_error_rate("the quick brown fox", "the quick fox") == pytest.approx(1 / 4)

    def test_repeated_whitespace_scores_zero(self):
        assert word_error_rate("the  quick   fox", "the quick fox") == 0.0

    def test_multiline_text_scores_zero(self):
        assert word_error_rate("the quick\nbrown fox", "the quick brown fox") == 0.0

    def test_empty_hypothesis_scores_one(self):
        assert word_error_rate("the quick fox", "") == pytest.approx(1.0)

    def test_empty_reference_and_hypothesis_scores_zero(self):
        assert word_error_rate("", "") == 0.0

    def test_empty_reference_with_output_scores_one(self):
        assert word_error_rate("", "spurious words") == 1.0

    def test_identical_arabic_scores_zero(self):
        assert word_error_rate(ARABIC_LINE, ARABIC_LINE) == 0.0

    def test_arabic_word_substitution_is_detected(self):
        words = ARABIC_LINE.split()
        altered = " ".join(["مختلف"] + words[1:])
        assert word_error_rate(ARABIC_LINE, altered) == pytest.approx(1 / len(words))


# --------------------------------------------------------------------------
# Fake engine + fixtures
# --------------------------------------------------------------------------


class _FakeEngine(OCREngine):
    """
    Fake OCR engine returning canned text, so evaluator tests never
    depend on Tesseract being installed.
    """

    name = "fake"

    def __init__(self, text="", confidence=88.0, raise_error=False):
        self._text = text
        self._confidence = confidence
        self._raise_error = raise_error

    def recognize_raw(self, image, language):  # pragma: no cover - not used
        raise NotImplementedError

    def process(self, image, language=None, config=None):
        from src.ocr.models import BoundingBox, PageOCRResult, WordResult

        if self._raise_error:
            raise OCRError("Simulated engine failure")

        words = [
            WordResult(text=w, confidence=self._confidence, bounding_box=BoundingBox(0, 0, 10, 10))
            for w in self._text.split()
        ]
        return PageOCRResult(
            page_number=1,
            text=self._text,
            language=language or "eng",
            engine=self.name,
            words=words,
            mean_confidence=self._confidence if words else 0.0,
            processing_time_ms=12.5,
        )


@pytest.fixture
def eval_dir(tmp_path: Path) -> Path:
    """Build a tiny on-disk evaluation dataset."""
    (tmp_path / "images").mkdir()
    (tmp_path / "ground_truth").mkdir()

    image = np.full((60, 200, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "images" / "s1.png"), image)
    cv2.imwrite(str(tmp_path / "images" / "s2.png"), image)

    (tmp_path / "ground_truth" / "s1.txt").write_text("hello world", encoding="utf-8")
    (tmp_path / "ground_truth" / "s2.txt").write_text("second sample", encoding="utf-8")

    manifest = {
        "samples": [
            {
                "id": "s1",
                "image": "images/s1.png",
                "ground_truth": "ground_truth/s1.txt",
                "language": "eng",
                "category": "clean",
            },
            {
                "id": "s2",
                "image": "images/s2.png",
                "ground_truth": "ground_truth/s2.txt",
                "language": "eng",
                "category": "blurry",
            },
        ]
    }
    (tmp_path / "dataset.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


def _sample(sample_id="s1", category="clean", image="images/s1.png", truth="ground_truth/s1.txt"):
    return EvaluationSample(
        id=sample_id, image=image, ground_truth=truth, language="eng", category=category
    )


# --------------------------------------------------------------------------
# Dataset loading
# --------------------------------------------------------------------------


class TestDatasetLoading:
    def test_loads_samples(self, eval_dir):
        samples = load_dataset(eval_dir / "dataset.json")
        assert len(samples) == 2
        assert samples[0].id == "s1"

    def test_missing_manifest_raises(self, tmp_path):
        with pytest.raises(EvaluationError, match="not found"):
            load_dataset(tmp_path / "nope.json")

    def test_invalid_json_raises(self, tmp_path):
        bad = tmp_path / "dataset.json"
        bad.write_text("{not json", encoding="utf-8")
        with pytest.raises(EvaluationError, match="valid JSON"):
            load_dataset(bad)

    def test_entry_missing_required_field_raises(self, tmp_path):
        bad = tmp_path / "dataset.json"
        bad.write_text(json.dumps({"samples": [{"id": "x"}]}), encoding="utf-8")
        with pytest.raises(EvaluationError, match="missing required field"):
            load_dataset(bad)

    def test_accepts_bare_list_manifest(self, tmp_path):
        manifest = [{"id": "a", "image": "i.png", "ground_truth": "g.txt"}]
        path = tmp_path / "dataset.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        assert len(load_dataset(path)) == 1


# --------------------------------------------------------------------------
# Single-sample evaluation
# --------------------------------------------------------------------------


class TestEvaluateSample:
    def test_perfect_ocr_scores_zero_error(self, eval_dir):
        result = evaluate_sample(_sample(), _FakeEngine("hello world"), eval_dir)
        assert result.success
        assert result.cer == 0.0
        assert result.wer == 0.0
        assert result.engine == "fake"

    def test_imperfect_ocr_scores_nonzero_error(self, eval_dir):
        result = evaluate_sample(_sample(), _FakeEngine("hello word"), eval_dir)
        assert result.success
        assert result.cer > 0.0
        assert result.wer > 0.0

    def test_captures_confidence_and_timing(self, eval_dir):
        result = evaluate_sample(_sample(), _FakeEngine("hello world"), eval_dir)
        assert result.ocr_confidence == pytest.approx(88.0)
        assert result.processing_time_ms == pytest.approx(12.5)

    def test_records_text_lengths_and_word_counts(self, eval_dir):
        result = evaluate_sample(_sample(), _FakeEngine("hello world"), eval_dir)
        assert result.reference_word_count == 2
        assert result.recognized_word_count == 2
        assert result.reference_text_length == len("hello world")

    def test_empty_ocr_output_is_scored_not_crashed(self, eval_dir):
        result = evaluate_sample(_sample(), _FakeEngine(""), eval_dir)
        assert result.success
        assert result.cer == pytest.approx(1.0)
        assert result.recognized_word_count == 0

    def test_engine_failure_is_captured_as_unsuccessful(self, eval_dir):
        result = evaluate_sample(_sample(), _FakeEngine(raise_error=True), eval_dir)
        assert not result.success
        assert "Simulated engine failure" in result.error
        # Metrics must stay None, never a placeholder zero.
        assert result.cer is None
        assert result.wer is None

    def test_missing_image_is_captured(self, eval_dir):
        result = evaluate_sample(_sample(image="images/nope.png"), _FakeEngine("x"), eval_dir)
        assert not result.success
        assert "Image not found" in result.error

    def test_missing_ground_truth_is_captured(self, eval_dir):
        result = evaluate_sample(_sample(truth="ground_truth/nope.txt"), _FakeEngine("x"), eval_dir)
        assert not result.success
        assert "Ground truth not found" in result.error

    def test_result_is_serializable(self, eval_dir):
        result = evaluate_sample(_sample(), _FakeEngine("hello world"), eval_dir)
        payload = result.to_dict()
        assert json.loads(json.dumps(payload))["dataset_id"] == "s1"


# --------------------------------------------------------------------------
# Summary statistics
# --------------------------------------------------------------------------


class TestSummarize:
    def _result(self, sample_id, cer, wer, category="clean", success=True, time_ms=10.0):
        return EvaluationResult(
            dataset_id=sample_id,
            engine="fake",
            language="eng",
            category=category,
            cer=cer if success else None,
            wer=wer if success else None,
            ocr_confidence=90.0 if success else None,
            processing_time_ms=time_ms if success else None,
            success=success,
        )

    def test_counts_successes_and_failures(self):
        results = [self._result("a", 0.0, 0.0), self._result("b", None, None, success=False)]
        summary = summarize(results, "fake")
        assert summary.total_samples == 2
        assert summary.successful_samples == 1
        assert summary.failed_samples == 1

    def test_averages_and_medians(self):
        results = [self._result("a", 0.0, 0.0), self._result("b", 0.5, 1.0)]
        summary = summarize(results, "fake")
        assert summary.average_cer == pytest.approx(0.25)
        assert summary.median_cer == pytest.approx(0.25)
        assert summary.average_wer == pytest.approx(0.5)

    def test_failed_samples_do_not_count_as_zero_error(self):
        results = [self._result("a", 0.5, 0.5), self._result("b", None, None, success=False)]
        summary = summarize(results, "fake")
        # Average must be 0.5 (the one real sample), not 0.25.
        assert summary.average_cer == pytest.approx(0.5)

    def test_identifies_fastest_and_slowest(self):
        results = [self._result("a", 0.0, 0.0, time_ms=5.0), self._result("b", 0.0, 0.0, time_ms=50.0)]
        summary = summarize(results, "fake")
        assert summary.fastest_sample == "a"
        assert summary.slowest_sample == "b"

    def test_category_breakdown(self):
        results = [
            self._result("a", 0.0, 0.0, category="clean"),
            self._result("b", 0.4, 0.4, category="blurry"),
        ]
        summary = summarize(results, "fake")
        assert summary.by_category["clean"]["samples"] == 1
        assert summary.by_category["blurry"]["average_cer"] == pytest.approx(0.4)

    def test_all_failed_leaves_metrics_none(self):
        results = [self._result("a", None, None, success=False)]
        summary = summarize(results, "fake")
        assert summary.average_cer is None
        assert summary.successful_samples == 0

    def test_empty_results(self):
        summary = summarize([], "fake")
        assert summary.total_samples == 0
        assert summary.average_cer is None


# --------------------------------------------------------------------------
# Batch evaluation
# --------------------------------------------------------------------------


class TestBatchEvaluation:
    def test_evaluates_every_sample(self, eval_dir):
        results, summary = evaluate_dataset(
            dataset_path=eval_dir / "dataset.json",
            engine=_FakeEngine("hello world"),
            results_dir=None,
        )
        assert len(results) == 2
        assert summary.total_samples == 2

    def test_writes_result_files(self, eval_dir, tmp_path):
        out = tmp_path / "results"
        evaluate_dataset(
            dataset_path=eval_dir / "dataset.json",
            engine=_FakeEngine("hello world"),
            results_dir=out,
        )
        assert (out / "results.json").is_file()
        assert (out / "results.csv").is_file()

    def test_result_json_contains_summary_and_results(self, eval_dir, tmp_path):
        out = tmp_path / "results"
        evaluate_dataset(
            dataset_path=eval_dir / "dataset.json",
            engine=_FakeEngine("hello world"),
            results_dir=out,
        )
        payload = json.loads((out / "results.json").read_text(encoding="utf-8"))
        assert "summary" in payload
        assert len(payload["results"]) == 2

    def test_failed_samples_do_not_abort_the_run(self, eval_dir):
        results, summary = evaluate_dataset(
            dataset_path=eval_dir / "dataset.json",
            engine=_FakeEngine(raise_error=True),
            results_dir=None,
        )
        assert len(results) == 2
        assert summary.failed_samples == 2
        assert summary.successful_samples == 0

    def test_save_results_creates_directory(self, tmp_path):
        result = EvaluationResult(
            dataset_id="x", engine="fake", language="eng", category="clean", success=True, cer=0.0, wer=0.0
        )
        summary = summarize([result], "fake")
        paths = save_results([result], summary, tmp_path / "nested" / "results")
        assert paths["json"].is_file()
        assert paths["csv"].is_file()

    def test_arabic_survives_the_json_round_trip(self, tmp_path):
        result = EvaluationResult(
            dataset_id=ARABIC_LINE, engine="fake", language="ara", category="arabic", success=True
        )
        summary = summarize([result], "fake")
        paths = save_results([result], summary, tmp_path / "results")
        payload = json.loads(paths["json"].read_text(encoding="utf-8"))
        assert payload["results"][0]["dataset_id"] == ARABIC_LINE
