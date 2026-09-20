"""
Tests for Phase 2: image quality assessment.

All test images are generated synthetically with NumPy/OpenCV so the
suite has no external file dependencies. Assertions check qualitative
direction and classification (e.g. "sharp scores higher than blurry")
rather than exact floating-point values, since these are heuristic
classical-CV metrics whose precise magnitudes aren't meaningful to pin down.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.quality.image_quality import (
    BlurStatus,
    ContrastStatus,
    ImageQualityConfig,
    ImageQualityError,
    NoiseStatus,
    OverallQuality,
    PerspectiveStatus,
    PreprocessingOperation,
    SkewStatus,
    assess_blur,
    assess_contrast,
    assess_noise,
    assess_page_quality,
    assess_page_quality_safe,
    assess_perspective,
    assess_resolution,
    assess_skew,
)


# --------------------------------------------------------------------------
# Synthetic image builders
# --------------------------------------------------------------------------


def _make_text_like_page(width: int = 1200, height: int = 1600) -> np.ndarray:
    """
    A clean, high-contrast page with sharp black text-line-like strokes.

    Lines are horizontal, left-aligned, and of varying length (like real
    text lines of different lengths) rather than a symmetric grid. A
    symmetric grid (equal horizontal + vertical strokes) is a poor proxy
    for real scanned text: its near-square bounding shape makes
    rectangle-based skew estimation ambiguous/unstable. Real text has a
    clear dominant horizontal orientation, which this mimics.
    """
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
    # Compress the dynamic range into a narrow mid-gray band.
    compressed = (128 + (gray.astype(np.float64) - 128) * 0.05).astype(np.uint8)
    return cv2.cvtColor(compressed, cv2.COLOR_GRAY2RGB)


def _make_noisy_version(image: np.ndarray, sigma: float = 35.0) -> np.ndarray:
    noise = np.random.default_rng(42).normal(0, sigma, image.shape)
    noisy = np.clip(image.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    return noisy


def _make_rotated_version(image: np.ndarray, angle_degrees: float) -> np.ndarray:
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    rotated = cv2.warpAffine(
        image, matrix, (width, height), borderValue=(255, 255, 255)
    )
    return rotated


def _make_low_resolution_page() -> np.ndarray:
    small = np.full((150, 200), 255, dtype=np.uint8)
    cv2.line(small, (10, 20), (190, 20), color=0, thickness=1)
    return cv2.cvtColor(small, cv2.COLOR_GRAY2RGB)


def _make_photographed_document(
    quad: np.ndarray = None, bg_size=(900, 1200)
) -> np.ndarray:
    """
    A synthetic "phone photo of a document" page: a light, roughly
    document-colored quadrilateral (not axis-aligned, mimicking a
    photographed-at-an-angle page) on a larger noisy/textured dark
    background, with a few horizontal strokes inside standing in for text.
    """
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


def _make_full_frame_page() -> np.ndarray:
    """A page whose black border quadrilateral fills almost the entire frame."""
    width, height = 1000, 1300
    page = np.full((height, width, 3), 235, dtype=np.uint8)
    cv2.rectangle(page, (10, 10), (width - 10, height - 10), (10, 10, 10), 6)
    for i in range(6):
        y = 150 + i * 150
        cv2.line(page, (60, y), (width - 60, y), (20, 20, 20), 3)
    return page


# --------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------


class TestResolution:
    def test_known_high_dpi_is_not_low_resolution(self):
        image = _make_text_like_page()
        result = assess_resolution(image, dpi=300)
        assert result.is_low_resolution is False
        assert result.dpi == 300
        assert result.total_pixels == image.shape[0] * image.shape[1]

    def test_known_low_dpi_is_flagged(self):
        image = _make_text_like_page()
        result = assess_resolution(image, dpi=72)
        assert result.is_low_resolution is True

    def test_missing_dpi_falls_back_to_pixel_dimensions(self):
        small_image = _make_low_resolution_page()
        result = assess_resolution(small_image, dpi=None)
        assert result.dpi is None
        assert result.is_low_resolution is True

    def test_missing_dpi_large_image_not_flagged(self):
        image = _make_text_like_page(width=2000, height=2600)
        result = assess_resolution(image, dpi=None)
        assert result.is_low_resolution is False


# --------------------------------------------------------------------------
# Blur
# --------------------------------------------------------------------------


class TestBlur:
    def test_sharp_image_classified_sharp(self):
        image = _make_text_like_page()
        result = assess_blur(image)
        assert result.status is BlurStatus.SHARP

    def test_blurry_image_scores_lower_than_sharp(self):
        sharp_image = _make_text_like_page()
        blurry_image = _make_blurry_version(sharp_image)

        sharp_result = assess_blur(sharp_image)
        blurry_result = assess_blur(blurry_image)

        assert blurry_result.score < sharp_result.score
        assert blurry_result.status in (BlurStatus.BLURRY, BlurStatus.MODERATELY_BLURRY)

    def test_thresholds_are_configurable(self):
        image = _make_text_like_page()
        # An absurdly high sharp threshold forces even a sharp image to read as blurry.
        strict_config = ImageQualityConfig(blur_sharp_threshold=1e9, blur_blurry_threshold=1e8)
        result = assess_blur(image, config=strict_config)
        assert result.status is BlurStatus.BLURRY


# --------------------------------------------------------------------------
# Skew
# --------------------------------------------------------------------------


class TestSkew:
    def test_unrotated_page_is_acceptable(self):
        image = _make_text_like_page()
        result = assess_skew(image)
        assert result.status is SkewStatus.ACCEPTABLE
        assert abs(result.angle_degrees) < 2.0

    def test_significantly_rotated_page_flagged(self):
        image = _make_text_like_page()
        rotated = _make_rotated_version(image, angle_degrees=8.0)
        result = assess_skew(rotated)
        assert result.status is SkewStatus.NEEDS_DESKEW
        assert abs(result.angle_degrees) > 2.0

    def test_slightly_rotated_page_may_be_acceptable(self):
        image = _make_text_like_page()
        rotated = _make_rotated_version(image, angle_degrees=0.3)
        result = assess_skew(rotated)
        assert result.status is SkewStatus.ACCEPTABLE

    def test_blank_page_defaults_to_zero_skew(self):
        blank = np.full((500, 500, 3), 255, dtype=np.uint8)
        result = assess_skew(blank)
        assert result.angle_degrees == 0.0
        assert result.status is SkewStatus.ACCEPTABLE


# --------------------------------------------------------------------------
# Perspective
# --------------------------------------------------------------------------


class TestPerspective:
    def test_photographed_document_needs_correction(self):
        image = _make_photographed_document()
        result = assess_perspective(image)
        assert result.status is PerspectiveStatus.NEEDS_CORRECTION
        assert result.corners is not None
        assert len(result.corners) == 4

    def test_detected_corners_are_close_to_true_corners(self):
        quad = np.array([[250, 150], [950, 100], [1000, 800], [200, 850]], dtype=np.int32)
        image = _make_photographed_document(quad=quad)
        result = assess_perspective(image)
        assert result.corners is not None
        # true_corners is already in (top-left, top-right, bottom-right,
        # bottom-left) order since that's how the quad was constructed.
        true_corners = quad.astype(np.float64)
        for detected, true in zip(result.corners, true_corners):
            assert abs(detected[0] - true[0]) < 15
            assert abs(detected[1] - true[1]) < 15

    def test_full_frame_page_not_needed(self):
        image = _make_full_frame_page()
        result = assess_perspective(image)
        assert result.status is PerspectiveStatus.NOT_NEEDED
        assert result.frame_coverage_ratio > 0.9

    def test_no_quadrilateral_returns_undetected(self):
        # Mild texture/noise with no strong edges or shapes anywhere —
        # unlike harsh full-range noise, which can accidentally produce
        # dense edges that approximate the whole frame as "a quad."
        rng = np.random.default_rng(11)
        base = np.full((400, 400, 3), 200, dtype=np.uint8)
        noise = rng.normal(0, 6, (400, 400, 3))
        textured = np.clip(base.astype(np.float64) + noise, 0, 255).astype(np.uint8)
        result = assess_perspective(textured)
        assert result.status is PerspectiveStatus.UNDETECTED
        assert result.corners is None

    def test_blank_page_returns_undetected(self):
        blank = np.full((400, 400, 3), 255, dtype=np.uint8)
        result = assess_perspective(blank)
        assert result.status is PerspectiveStatus.UNDETECTED
        assert result.corners is None

    def test_coverage_ratio_is_configurable(self):
        image = _make_photographed_document()
        # Raising the minimum coverage above what this fixture achieves
        # (~49%) should make it fall back to UNDETECTED.
        strict_config = ImageQualityConfig(perspective_min_coverage_ratio=0.9)
        result = assess_perspective(image, config=strict_config)
        assert result.status is PerspectiveStatus.UNDETECTED


    def test_inner_rectangle_does_not_beat_outer_document(self):
        """A large inner table must not be selected over the page boundary."""
        image = _make_photographed_document(
            quad=np.array(
                [[120, 80], [1080, 110], [1050, 820], [140, 850]],
                dtype=np.int32,
            ),
            bg_size=(900, 1200),
        )
        # Add a strong inner rectangle that is deliberately easy to detect.
        cv2.rectangle(image, (300, 250), (900, 650), (10, 10, 10), 12)

        result = assess_perspective(image)
        assert result.status is PerspectiveStatus.NEEDS_CORRECTION
        assert result.corners is not None

        corners = np.asarray(result.corners)
        assert corners[:, 0].min() < 180
        assert corners[:, 1].min() < 130
        assert corners[:, 0].max() > 1020
        assert corners[:, 1].max() > 760

    def test_frame_merged_artifact_is_rejected_when_page_candidate_exists(self):
        """A contour fused to opposite frame borders must not win by area."""
        height, width = 900, 1200
        image = np.full((height, width, 3), 70, dtype=np.uint8)

        # Real page: large, trapezoidal, but clearly inside the frame.
        page = np.array(
            [[180, 100], [1030, 80], [1100, 790], [150, 820]],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(image, page, (225, 220, 205))

        # Strong frame lines create the same kind of merged contour artifact
        # that caused the real photographed certificate to fail detection.
        cv2.rectangle(image, (100, 2), (1100, height - 3), (5, 5, 5), 8)

        result = assess_perspective(image)
        assert result.status is PerspectiveStatus.NEEDS_CORRECTION
        assert result.corners is not None

        corners = np.asarray(result.corners)
        assert corners[:, 0].min() > 100
        assert corners[:, 0].max() < 1150
        assert corners[:, 1].min() > 50
        assert corners[:, 1].max() < 850


# --------------------------------------------------------------------------
# Contrast
# --------------------------------------------------------------------------


class TestContrast:
    def test_high_contrast_page_classified_good(self):
        image = _make_text_like_page()
        result = assess_contrast(image)
        assert result.status is ContrastStatus.GOOD

    def test_low_contrast_page_scores_lower_and_flagged(self):
        sharp_image = _make_text_like_page()
        low_contrast_image = _make_low_contrast_version(sharp_image)

        good_result = assess_contrast(sharp_image)
        low_result = assess_contrast(low_contrast_image)

        assert low_result.score < good_result.score
        assert low_result.status is ContrastStatus.LOW


# --------------------------------------------------------------------------
# Noise
# --------------------------------------------------------------------------


class TestNoise:
    def test_clean_page_has_low_noise(self):
        image = _make_text_like_page()
        result = assess_noise(image)
        assert result.status is NoiseStatus.LOW

    def test_noisy_page_scores_higher_than_clean(self):
        clean_image = _make_text_like_page()
        noisy_image = _make_noisy_version(clean_image)

        clean_result = assess_noise(clean_image)
        noisy_result = assess_noise(noisy_image)

        assert noisy_result.score > clean_result.score
        assert noisy_result.status in (NoiseStatus.MODERATE, NoiseStatus.HIGH)


# --------------------------------------------------------------------------
# Full page assessment + recommendations
# --------------------------------------------------------------------------


class TestPageQualityAssessment:
    def test_clean_page_requires_no_preprocessing(self):
        image = _make_text_like_page()
        report = assess_page_quality(image, page_number=1, dpi=300)

        assert report.overall_quality in (OverallQuality.EXCELLENT, OverallQuality.GOOD)
        assert report.recommended_operations == []

    def test_blurry_page_recommends_denoise(self):
        image = _make_blurry_version(_make_text_like_page(), ksize=31)
        report = assess_page_quality(image, page_number=2, dpi=300)

        assert PreprocessingOperation.DENOISE.value in report.recommended_operations
        assert report.overall_quality is not OverallQuality.EXCELLENT

    def test_rotated_page_recommends_deskew(self):
        image = _make_rotated_version(_make_text_like_page(), angle_degrees=6.0)
        report = assess_page_quality(image, page_number=3, dpi=300)

        assert PreprocessingOperation.DESKEW.value in report.recommended_operations

    def test_low_contrast_page_recommends_contrast_enhancement(self):
        image = _make_low_contrast_version(_make_text_like_page())
        report = assess_page_quality(image, page_number=4, dpi=300)

        assert PreprocessingOperation.CONTRAST_ENHANCEMENT.value in report.recommended_operations

    def test_noisy_page_recommends_denoise(self):
        image = _make_noisy_version(_make_text_like_page())
        report = assess_page_quality(image, page_number=5, dpi=300)

        assert PreprocessingOperation.DENOISE.value in report.recommended_operations

    def test_low_resolution_page_recommends_resize(self):
        image = _make_low_resolution_page()
        report = assess_page_quality(image, page_number=6, dpi=None)

        assert PreprocessingOperation.RESIZE.value in report.recommended_operations

    def test_combination_of_problems_is_poor_with_multiple_recommendations(self):
        image = _make_text_like_page()
        image = _make_blurry_version(image, ksize=31)
        image = _make_low_contrast_version(image)
        image = _make_rotated_version(image, angle_degrees=7.0)

        report = assess_page_quality(image, page_number=7, dpi=72)

        assert report.overall_quality is OverallQuality.POOR
        assert len(report.recommended_operations) >= 3

    def test_every_recommendation_has_a_reason(self):
        image = _make_low_contrast_version(_make_blurry_version(_make_text_like_page()))
        report = assess_page_quality(image, page_number=8, dpi=300)

        assert len(report.recommendations) > 0
        for recommendation in report.recommendations:
            assert recommendation.reason  # non-empty explanation
            assert isinstance(recommendation.reason, str)

    def test_to_dict_matches_expected_shape(self):
        image = _make_text_like_page()
        report = assess_page_quality(image, page_number=1, dpi=300)
        payload = report.to_dict()

        for key in (
            "page_number",
            "width",
            "height",
            "dpi",
            "blur_score",
            "blur_status",
            "skew_angle",
            "skew_status",
            "contrast_score",
            "contrast_status",
            "noise_score",
            "noise_status",
            "overall_quality",
            "recommended_operations",
        ):
            assert key in payload


# --------------------------------------------------------------------------
# Error handling
# --------------------------------------------------------------------------


class TestErrorHandling:
    def test_none_image_raises(self):
        with pytest.raises(ImageQualityError):
            assess_page_quality(None, page_number=1)

    def test_empty_array_raises(self):
        with pytest.raises(ImageQualityError):
            assess_page_quality(np.array([]), page_number=1)

    def test_wrong_type_raises(self):
        with pytest.raises(ImageQualityError):
            assess_page_quality("not an image", page_number=1)  # type: ignore[arg-type]

    def test_safe_variant_never_raises_and_returns_none_on_failure(self):
        result = assess_page_quality_safe(None, page_number=1)
        assert result is None

    def test_safe_variant_returns_report_on_valid_input(self):
        image = _make_text_like_page()
        result = assess_page_quality_safe(image, page_number=1, dpi=300)
        assert result is not None
        assert result.page_number == 1
