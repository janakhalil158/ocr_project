"""
Tests for Phase 3: document image preprocessing.

Follows the same approach as ``test_image_quality.py``: all test images
are generated synthetically with NumPy/OpenCV, and assertions check
qualitative/structural behavior (was it resized, is it straighter, did
dimensions survive) rather than exact pixel values, since these are
classical CV transforms whose precise output isn't meaningful to pin
down to the pixel.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.preprocessing.preprocessing import (
    PreprocessingConfig,
    PreprocessingError,
    ProcessedPage,
    correct_perspective,
    deskew_image,
    denoise_image,
    enhance_contrast,
    preprocess_page,
    resize_image,
    threshold_image,
)
from src.quality.image_quality import (
    ImageQualityConfig,
    PreprocessingOperation,
    assess_page_quality,
    assess_perspective,
    assess_skew,
)

# --------------------------------------------------------------------------
# Synthetic image builders (mirrors test_image_quality.py's builders so
# the two suites exercise Phase 2 and Phase 3 against comparable inputs)
# --------------------------------------------------------------------------


def _make_text_like_page(width: int = 1200, height: int = 1600) -> np.ndarray:
    rng = np.random.default_rng(7)
    page = np.full((height, width), 255, dtype=np.uint8)
    for y in range(100, height - 100, 35):
        line_length = int(rng.uniform(0.4, 1.0) * (width - 160))
        cv2.line(page, (80, y), (80 + line_length, y), color=0, thickness=4)
    return cv2.cvtColor(page, cv2.COLOR_GRAY2RGB)


def _make_blurry_version(image: np.ndarray, ksize: int = 25) -> np.ndarray:
    return cv2.GaussianBlur(image, (ksize, ksize), 0)


def _make_low_contrast_version(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    compressed = (128 + (gray.astype(np.float64) - 128) * 0.05).astype(np.uint8)
    return cv2.cvtColor(compressed, cv2.COLOR_GRAY2RGB)


def _make_noisy_version(image: np.ndarray, sigma: float = 35.0) -> np.ndarray:
    noise = np.random.default_rng(42).normal(0, sigma, image.shape)
    return np.clip(image.astype(np.float64) + noise, 0, 255).astype(np.uint8)


def _make_rotated_version(image: np.ndarray, angle_degrees: float) -> np.ndarray:
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    return cv2.warpAffine(image, matrix, (width, height), borderValue=(255, 255, 255))


def _make_low_resolution_page() -> np.ndarray:
    small = np.full((150, 200), 255, dtype=np.uint8)
    cv2.line(small, (10, 20), (190, 20), color=0, thickness=1)
    return cv2.cvtColor(small, cv2.COLOR_GRAY2RGB)


def _make_photographed_document(quad: np.ndarray = None, bg_size=(900, 1200)) -> np.ndarray:
    """A light, trapezoidal (photographed-at-an-angle) page on a larger noisy background."""
    height, width = bg_size
    rng = np.random.default_rng(3)
    background = rng.integers(30, 90, (height, width, 3), dtype=np.uint8)
    if quad is None:
        quad = np.array([[250, 150], [950, 100], [1000, 800], [200, 850]], dtype=np.int32)
    cv2.fillConvexPoly(background, quad, (235, 225, 200))
    for i in range(5):
        y = 250 + i * 80
        cv2.line(background, (320, y), (850, y), (20, 20, 20), 4)
    return background


# --------------------------------------------------------------------------
# Resize
# --------------------------------------------------------------------------


class TestResize:
    def test_low_dpi_page_is_upscaled(self):
        image = _make_text_like_page()
        result = resize_image(image, current_dpi=72)
        assert result.shape[0] > image.shape[0]
        assert result.shape[1] > image.shape[1]

    def test_adequate_dpi_page_is_not_resized(self):
        image = _make_text_like_page()
        result = resize_image(image, current_dpi=300)
        assert result is image  # unchanged, no unnecessary copy

    def test_missing_dpi_falls_back_to_pixel_dimensions(self):
        image = _make_low_resolution_page()
        result = resize_image(image, current_dpi=None)
        assert result.shape[0] > image.shape[0]
        assert result.shape[1] > image.shape[1]

    def test_adequate_pixel_dimensions_not_resized(self):
        image = _make_text_like_page(width=2000, height=2600)
        result = resize_image(image, current_dpi=None)
        assert result is image

    def test_aspect_ratio_is_preserved(self):
        image = _make_low_resolution_page()  # 200x150 -> ratio 4:3
        result = resize_image(image, current_dpi=None)
        original_ratio = image.shape[1] / image.shape[0]
        new_ratio = result.shape[1] / result.shape[0]
        assert new_ratio == pytest.approx(original_ratio, rel=1e-2)

    def test_output_dimensions_are_valid(self):
        image = _make_low_resolution_page()
        result = resize_image(image, current_dpi=None)
        assert result.ndim == image.ndim
        assert result.dtype == np.uint8
        assert result.shape[0] > 0 and result.shape[1] > 0

    def test_upscale_is_capped(self):
        tiny = np.full((10, 10, 3), 255, dtype=np.uint8)
        cv2.line(tiny, (1, 5), (8, 5), color=0, thickness=1)
        config = PreprocessingConfig(max_upscale_factor=2.0)
        result = resize_image(tiny, current_dpi=None, config=config)
        assert result.shape[0] <= tiny.shape[0] * 2.0 + 1
        assert result.shape[1] <= tiny.shape[1] * 2.0 + 1


# --------------------------------------------------------------------------
# Perspective correction
# --------------------------------------------------------------------------


class TestPerspectiveCorrection:
    def test_none_corners_returns_image_unchanged(self):
        image = _make_photographed_document()
        result = correct_perspective(image, corners=None)
        assert result is image

    def test_wrong_number_of_corners_raises(self):
        image = _make_photographed_document()
        with pytest.raises(PreprocessingError):
            correct_perspective(image, corners=[(0, 0), (1, 0), (1, 1)])

    def test_degenerate_corners_returns_image_unchanged(self):
        image = _make_photographed_document()
        # All 4 points collapsed to (near) the same spot -> zero-area output.
        result = correct_perspective(image, corners=[(0, 0), (1, 0), (1, 1), (0, 1)])
        assert result is image

    def test_valid_quad_produces_straight_rectangle(self):
        quad = np.array([[250, 150], [950, 100], [1000, 800], [200, 850]], dtype=np.int32)
        image = _make_photographed_document(quad=quad)
        metrics = assess_perspective(image)
        assert metrics.corners is not None

        result = correct_perspective(image, corners=metrics.corners)
        assert result.dtype == np.uint8
        # Background (dark, ~30-90 range) should be almost entirely gone;
        # the warped output should be dominated by the light page color.
        gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
        assert float(np.mean(gray > 150)) > 0.7

    def test_output_size_matches_corner_geometry(self):
        # A perfect axis-aligned rectangle's corners should produce output
        # dimensions matching its own width/height almost exactly.
        image = np.full((600, 800, 3), 220, dtype=np.uint8)
        corners = [(100.0, 50.0), (700.0, 50.0), (700.0, 550.0), (100.0, 550.0)]
        result = correct_perspective(image, corners=corners)
        assert result.shape[1] == pytest.approx(600, abs=2)
        assert result.shape[0] == pytest.approx(500, abs=2)

    def test_invalid_input_raises_preprocessing_error(self):
        with pytest.raises(PreprocessingError):
            correct_perspective(None, corners=None)


# --------------------------------------------------------------------------
# Denoise
# --------------------------------------------------------------------------


class TestDenoise:
    def test_noisy_page_can_be_denoised(self):
        image = _make_text_like_page()
        noisy = _make_noisy_version(image)
        denoised = denoise_image(noisy)

        # Denoising should move the image back toward the clean original,
        # i.e. reduce the mean absolute difference versus the noisy input.
        diff_before = np.abs(noisy.astype(np.float64) - image.astype(np.float64)).mean()
        diff_after = np.abs(denoised.astype(np.float64) - image.astype(np.float64)).mean()
        assert diff_after < diff_before

    def test_dimensions_unchanged(self):
        image = _make_noisy_version(_make_text_like_page())
        result = denoise_image(image)
        assert result.shape == image.shape

    def test_output_type_valid(self):
        image = _make_noisy_version(_make_text_like_page())
        result = denoise_image(image)
        assert result.dtype == np.uint8

    def test_grayscale_input_supported(self):
        image = cv2.cvtColor(_make_noisy_version(_make_text_like_page()), cv2.COLOR_RGB2GRAY)
        result = denoise_image(image)
        assert result.shape == image.shape
        assert result.ndim == 2

    def test_text_structure_not_destroyed(self):
        """A denoised page should still have strong, sharp edges (i.e. text
        survives), not be smoothed into a flat, near-uniform image."""
        image = _make_noisy_version(_make_text_like_page())
        result = denoise_image(image)
        gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
        assert gray.std() > 20  # still has real light/dark structure


# --------------------------------------------------------------------------
# Deskew
# --------------------------------------------------------------------------


class TestDeskew:
    def test_zero_degree_page_unchanged(self):
        image = _make_text_like_page()
        result = deskew_image(image, angle_degrees=0.0)
        assert result is image

    def test_already_straight_page_not_rotated(self):
        image = _make_text_like_page()
        result = deskew_image(image, angle_degrees=0.02)
        assert result is image  # below the negligible-angle floor

    def test_slight_positive_skew_corrected(self):
        image = _make_text_like_page()
        rotated = _make_rotated_version(image, angle_degrees=3.0)
        measured = assess_skew(rotated)
        corrected = deskew_image(rotated, angle_degrees=measured.angle_degrees)
        result = assess_skew(corrected)
        assert abs(result.angle_degrees) < abs(measured.angle_degrees)

    def test_significant_positive_skew_corrected(self):
        image = _make_text_like_page()
        rotated = _make_rotated_version(image, angle_degrees=15.0)
        measured = assess_skew(rotated)
        corrected = deskew_image(rotated, angle_degrees=measured.angle_degrees)
        result = assess_skew(corrected)
        assert abs(result.angle_degrees) < 2.0

    def test_negative_skew_corrected(self):
        image = _make_text_like_page()
        rotated = _make_rotated_version(image, angle_degrees=-8.0)
        measured = assess_skew(rotated)
        assert measured.angle_degrees > 0  # sanity: opposite sign of applied rotation
        corrected = deskew_image(rotated, angle_degrees=measured.angle_degrees)
        result = assess_skew(corrected)
        assert abs(result.angle_degrees) < 2.0

    def test_dimensions_preserved(self):
        image = _make_text_like_page()
        rotated = _make_rotated_version(image, angle_degrees=8.0)
        corrected = deskew_image(rotated, angle_degrees=8.0)
        assert corrected.shape == rotated.shape


# --------------------------------------------------------------------------
# Contrast
# --------------------------------------------------------------------------


class TestContrastEnhancement:
    def test_low_contrast_page_is_improved(self):
        image = _make_low_contrast_version(_make_text_like_page())
        result = enhance_contrast(image)
        gray_before = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray_after = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
        assert gray_after.std() > gray_before.std()

    def test_good_contrast_page_output_valid(self):
        image = _make_text_like_page()
        result = enhance_contrast(image)
        assert result.shape == image.shape
        assert result.dtype == np.uint8

    def test_grayscale_input_supported(self):
        image = cv2.cvtColor(_make_low_contrast_version(_make_text_like_page()), cv2.COLOR_RGB2GRAY)
        result = enhance_contrast(image)
        assert result.shape == image.shape
        assert result.ndim == 2


# --------------------------------------------------------------------------
# Thresholding
# --------------------------------------------------------------------------


class TestThresholding:
    def test_document_page_binarized(self):
        image = _make_text_like_page()
        result = threshold_image(image)
        assert set(np.unique(result)).issubset({0, 255})

    def test_output_valid(self):
        image = _make_text_like_page()
        result = threshold_image(image)
        assert result.dtype == np.uint8
        assert result.shape[:2] == image.shape[:2]

    def test_foreground_and_background_distinguishable(self):
        image = _make_text_like_page()
        result = threshold_image(image)
        # A real page has both ink (minority) and background (majority)
        # pixels after binarization, not a single uniform value.
        values, counts = np.unique(result, return_counts=True)
        assert len(values) == 2
        assert min(counts) > 0

    def test_adaptive_method_also_binarizes(self):
        image = _make_text_like_page()
        config = PreprocessingConfig(threshold_method="adaptive")
        result = threshold_image(image, config=config)
        assert set(np.unique(result)).issubset({0, 255})

    def test_unknown_method_raises(self):
        image = _make_text_like_page()
        config = PreprocessingConfig(threshold_method="bogus")
        with pytest.raises(PreprocessingError):
            threshold_image(image, config=config)


# --------------------------------------------------------------------------
# Pipeline (preprocess_page)
# --------------------------------------------------------------------------


class TestPreprocessPage:
    def test_single_recommendation_applies_only_that_operation(self):
        image = _make_rotated_version(_make_text_like_page(), angle_degrees=6.0)
        report = assess_page_quality(image, page_number=1, dpi=300)
        assert report.recommended_operations == [PreprocessingOperation.DESKEW.value]

        result = preprocess_page(image, report)
        assert result.operations_applied == [PreprocessingOperation.DESKEW]

    def test_photographed_document_gets_perspective_corrected(self):
        image = _make_photographed_document()
        report = assess_page_quality(image, page_number=1, dpi=300)
        assert PreprocessingOperation.PERSPECTIVE_CORRECTION.value in report.recommended_operations

        result = preprocess_page(image, report)
        assert PreprocessingOperation.PERSPECTIVE_CORRECTION in result.operations_applied
        # Cropped to the page: output should be smaller than the original
        # frame, and dominated by the light page color rather than the
        # dark background.
        assert result.image.shape[0] < image.shape[0]
        assert result.image.shape[1] < image.shape[1]
        gray = cv2.cvtColor(result.image, cv2.COLOR_RGB2GRAY)
        assert float(np.mean(gray > 130)) > 0.6

    def test_multiple_recommendations_all_applied(self):
        image = _make_text_like_page()
        image = _make_blurry_version(image, ksize=31)
        image = _make_low_contrast_version(image)
        image = _make_rotated_version(image, angle_degrees=7.0)
        report = assess_page_quality(image, page_number=2, dpi=72)
        assert len(report.recommended_operations) >= 3

        result = preprocess_page(image, report)
        for operation_name in report.recommended_operations:
            assert PreprocessingOperation(operation_name) in result.operations_applied

    def test_no_recommendations_returns_image_unmodified(self):
        image = _make_text_like_page()
        report = assess_page_quality(image, page_number=3, dpi=300)
        assert report.recommended_operations == []

        result = preprocess_page(image, report)
        assert result.operations_applied == []
        assert result.image is image

    def test_operations_applied_in_intended_order(self):
        image = _make_text_like_page()
        image = _make_noisy_version(image)
        image = _make_low_contrast_version(image)
        image = _make_rotated_version(image, angle_degrees=6.0)
        report = assess_page_quality(image, page_number=4, dpi=72)

        result = preprocess_page(image, report)
        expected_order = [
            op
            for op in (
                PreprocessingOperation.RESIZE,
                PreprocessingOperation.DENOISE,
                PreprocessingOperation.DESKEW,
                PreprocessingOperation.CONTRAST_ENHANCEMENT,
                PreprocessingOperation.THRESHOLDING,
            )
            if op in result.operations_applied
        ]
        assert result.operations_applied == expected_order

    def test_operations_applied_reflects_actual_work_not_just_recommendation(self):
        # A page whose skew is right at Phase 2's threshold can round-trip
        # to a "recommended but negligible" angle that deskew_image itself
        # declines to rotate; operations_applied must not claim it happened.
        image = _make_text_like_page()
        report = assess_page_quality(image, page_number=5, dpi=300)
        result = preprocess_page(image, report)
        assert PreprocessingOperation.DESKEW not in result.operations_applied
        assert PreprocessingOperation.RESIZE not in result.operations_applied

    def test_already_good_page_not_unnecessarily_modified(self):
        image = _make_text_like_page()
        report = assess_page_quality(image, page_number=6, dpi=300)
        result = preprocess_page(image, report)
        assert np.array_equal(result.image, image)

    def test_blank_page_does_not_crash(self):
        blank = np.full((500, 500, 3), 255, dtype=np.uint8)
        report = assess_page_quality(blank, page_number=7, dpi=300)
        result = preprocess_page(blank, report)  # should not raise
        assert result.image.shape[:2] == blank.shape[:2]

    def test_invalid_input_raises_preprocessing_error(self):
        image = _make_text_like_page()
        report = assess_page_quality(image, page_number=8, dpi=300)
        with pytest.raises(PreprocessingError):
            preprocess_page(None, report)

    def test_result_is_processed_page_instance(self):
        image = _make_text_like_page()
        report = assess_page_quality(image, page_number=9, dpi=300)
        result = preprocess_page(image, report)
        assert isinstance(result, ProcessedPage)
        assert result.page_number == 9

    def test_to_dict_matches_expected_shape(self):
        image = _make_rotated_version(_make_text_like_page(), angle_degrees=6.0)
        report = assess_page_quality(image, page_number=10, dpi=300)
        result = preprocess_page(image, report)
        payload = result.to_dict()
        for key in ("page_number", "width", "height", "channels", "operations_applied"):
            assert key in payload
