"""
Phase 2: Image quality assessment for scanned document pages.

Given a rendered page image, this module MEASURES resolution, blur,
skew, contrast, and noise using classical, explainable computer-vision
techniques, and RECOMMENDS which preprocessing operations may help —
without performing any preprocessing itself. Deskewing, denoising,
thresholding, etc. are all Phase 3+ concerns.

Design goals:
    * Every recommendation is explainable: it names the measurement
      and the configured threshold it crossed.
    * Every measurement function accepts a single page image and has
      no knowledge of "documents" or "batches" — this keeps the door
      open to processing millions of pages independently later
      (one page in, one report out) without rewriting this module.
    * No deep learning; all metrics are cheap, classical CV techniques
      appropriate for a first pass over large document volumes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.utils.config import IMAGE_QUALITY_CONFIG, ImageQualityConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ImageQualityError(Exception):
    """Raised when a page image cannot be assessed at all (e.g. unreadable, empty, or corrupt)."""


# --------------------------------------------------------------------------
# Status enums
# --------------------------------------------------------------------------


class BlurStatus(str, Enum):
    SHARP = "SHARP"
    MODERATELY_BLURRY = "MODERATELY_BLURRY"
    BLURRY = "BLURRY"


class SkewStatus(str, Enum):
    ACCEPTABLE = "ACCEPTABLE"
    NEEDS_DESKEW = "NEEDS_DESKEW"


class PerspectiveStatus(str, Enum):
    NOT_NEEDED = "NOT_NEEDED"  # a quadrilateral was found and it already fills the frame
    NEEDS_CORRECTION = "NEEDS_CORRECTION"  # a smaller quadrilateral was found within the frame
    UNDETECTED = "UNDETECTED"  # no quadrilateral could be reliably identified


class ContrastStatus(str, Enum):
    GOOD = "GOOD"
    MODERATE = "MODERATE"
    LOW = "LOW"


class NoiseStatus(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"


class OverallQuality(str, Enum):
    EXCELLENT = "EXCELLENT"
    GOOD = "GOOD"
    FAIR = "FAIR"
    POOR = "POOR"


class PreprocessingOperation(str, Enum):
    NONE = "NONE"
    RESIZE = "RESIZE"
    PERSPECTIVE_CORRECTION = "PERSPECTIVE_CORRECTION"
    DENOISE = "DENOISE"
    DESKEW = "DESKEW"
    CONTRAST_ENHANCEMENT = "CONTRAST_ENHANCEMENT"
    THRESHOLDING = "THRESHOLDING"


# --------------------------------------------------------------------------
# Per-metric result dataclasses
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolutionMetrics:
    """Resolution measurements for a page image."""

    width: int
    height: int
    total_pixels: int
    dpi: Optional[int]
    is_low_resolution: bool


@dataclass(frozen=True)
class BlurMetrics:
    """Blur/sharpness measurements, based on variance of the Laplacian.

    The variance of the Laplacian is a standard, cheap sharpness proxy:
    a sharp image has many strong high-frequency edges, so the Laplacian
    (a second-derivative edge operator) has high variance; a blurry
    image's edges are smoothed out, producing low variance. It's widely
    used as a fast, explainable focus/blur measure and correlates well
    with OCR-relevant edge sharpness, though it is resolution-dependent
    (rendering the same page at a higher DPI increases the score), so
    it should be tuned per rendering DPI rather than treated as universal.
    """

    score: float
    status: BlurStatus


@dataclass(frozen=True)
class SkewMetrics:
    """Estimated page rotation, in degrees, and whether it warrants deskewing."""

    angle_degrees: float
    status: SkewStatus


@dataclass(frozen=True)
class PerspectiveMetrics:
    """
    Detected document boundary within the frame, and whether perspective
    correction would help.

    Unlike the other Phase 2 metrics — which all assume the frame *is*
    the page and measure some property of it (how blurry, how skewed,
    how noisy) — this one first asks whether that assumption even
    holds: is the page the whole frame, or is it a photograph where the
    page is only part of a larger, cluttered scene (background, a hand
    holding it, etc.)? ``corners``, when present, are the four detected
    page corners in the original image, ordered (top-left, top-right,
    bottom-right, bottom-left).
    """

    corners: Optional[List[Tuple[float, float]]]
    frame_coverage_ratio: float
    status: PerspectiveStatus


@dataclass(frozen=True)
class ContrastMetrics:
    """Contrast measurement based on the standard deviation of grayscale intensities.

    Standard deviation of pixel intensity is a simple, well-understood
    global contrast metric: a page with mostly uniform gray values
    (washed-out scan) has low std deviation, while a page with a clean
    mix of dark ink and bright background has high std deviation.
    """

    score: float
    status: ContrastStatus


@dataclass(frozen=True)
class NoiseMetrics:
    """
    Noise estimate using the Immerkaer (1996) fast noise estimation method.

    This convolves the image with a Laplacian-like kernel designed so
    that it responds to noise while being largely insensitive to real
    image edges, then derives an approximate noise standard deviation
    from the response.

    LIMITATION: this is a heuristic estimate, not a ground-truth noise
    measurement. On text-dense scanned pages, fine glyph strokes can
    inflate the estimate somewhat, since they also produce high-frequency
    response. We deliberately do not disguise this as more precise than
    it is: treat `score` as a relative/comparative signal for triage
    (e.g. "this batch of pages is noisier than that batch"), and expect
    to recalibrate `noise_moderate_threshold` / `noise_high_threshold`
    against real, visually-reviewed documents before trusting it to
    drive automatic denoising decisions at scale.
    """

    score: float
    status: NoiseStatus


@dataclass(frozen=True)
class PreprocessingRecommendation:
    """A single recommended preprocessing operation with an explanation."""

    operation: PreprocessingOperation
    reason: str

    def to_dict(self) -> dict:
        return {"operation": self.operation.value, "reason": self.reason}


@dataclass(frozen=True)
class PageQualityReport:
    """Full Phase 2 quality report for a single page."""

    page_number: int
    resolution: ResolutionMetrics
    blur: BlurMetrics
    skew: SkewMetrics
    contrast: ContrastMetrics
    noise: NoiseMetrics
    perspective: PerspectiveMetrics
    overall_quality: OverallQuality
    recommendations: List[PreprocessingRecommendation] = field(default_factory=list)

    @property
    def recommended_operations(self) -> List[str]:
        """Flat list of recommended operation names, e.g. for JSON output."""
        return [r.operation.value for r in self.recommendations]

    def to_dict(self) -> dict:
        """Plain-dict representation matching the Phase 2 spec's example shape."""
        return {
            "page_number": self.page_number,
            "width": self.resolution.width,
            "height": self.resolution.height,
            "dpi": self.resolution.dpi,
            "blur_score": round(self.blur.score, 2),
            "blur_status": self.blur.status.value,
            "skew_angle": round(self.skew.angle_degrees, 2),
            "skew_status": self.skew.status.value,
            "contrast_score": round(self.contrast.score, 2),
            "contrast_status": self.contrast.status.value,
            "noise_score": round(self.noise.score, 4),
            "noise_status": self.noise.status.value,
            "perspective_status": self.perspective.status.value,
            "perspective_frame_coverage_ratio": round(self.perspective.frame_coverage_ratio, 4),
            "overall_quality": self.overall_quality.value,
            "recommended_operations": self.recommended_operations,
            "recommendation_details": [r.to_dict() for r in self.recommendations],
        }


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an RGB or grayscale image array to single-channel grayscale."""
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] in (3, 4):
        code = cv2.COLOR_RGB2GRAY if image.shape[2] == 3 else cv2.COLOR_RGBA2GRAY
        return cv2.cvtColor(image, code)
    raise ImageQualityError(f"Unsupported image shape for grayscale conversion: {image.shape}")


def _validate_image(image: np.ndarray) -> np.ndarray:
    """Validate that an array looks like a usable image; return it as grayscale."""
    if image is None:
        raise ImageQualityError("Received a None image.")
    if not isinstance(image, np.ndarray):
        raise ImageQualityError(f"Expected a numpy.ndarray, got {type(image)}.")
    if image.size == 0 or image.ndim not in (2, 3):
        raise ImageQualityError(f"Image has an invalid shape: {getattr(image, 'shape', None)}")

    gray = _to_grayscale(image)

    if gray.shape[0] < 3 or gray.shape[1] < 3:
        raise ImageQualityError(f"Image is too small to analyze: {gray.shape}")

    return gray


# --------------------------------------------------------------------------
# Individual metric functions
# --------------------------------------------------------------------------


def assess_resolution(
    image: np.ndarray,
    dpi: Optional[int] = None,
    config: ImageQualityConfig = IMAGE_QUALITY_CONFIG,
) -> ResolutionMetrics:
    """
    Measure resolution characteristics of a page image.

    DPI is treated as an external fact supplied by the caller (e.g. the
    DPI used to render the page from the PDF), NOT guessed from pixel
    data — pixel data alone cannot reliably reveal DPI. When DPI is
    unavailable, resolution adequacy falls back to a raw pixel-dimension
    floor.

    Args:
        image: Grayscale or RGB image array.
        dpi: Known rendering/scanning DPI, if available.
        config: Thresholds controlling the low-resolution rule.

    Returns:
        A :class:`ResolutionMetrics` instance.
    """
    gray = _validate_image(image)
    height, width = gray.shape[:2]
    total_pixels = width * height

    if dpi is not None:
        is_low_resolution = dpi < config.min_dpi
    else:
        is_low_resolution = width < config.min_width_px or height < config.min_height_px

    return ResolutionMetrics(
        width=width,
        height=height,
        total_pixels=total_pixels,
        dpi=dpi,
        is_low_resolution=is_low_resolution,
    )


def assess_blur(image: np.ndarray, config: ImageQualityConfig = IMAGE_QUALITY_CONFIG) -> BlurMetrics:
    """
    Measure blur/sharpness via the variance of the Laplacian.

    Args:
        image: Grayscale or RGB image array.
        config: Thresholds separating SHARP / MODERATELY_BLURRY / BLURRY.

    Returns:
        A :class:`BlurMetrics` instance.
    """
    gray = _validate_image(image)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    score = float(laplacian.var())

    if score >= config.blur_sharp_threshold:
        status = BlurStatus.SHARP
    elif score >= config.blur_blurry_threshold:
        status = BlurStatus.MODERATELY_BLURRY
    else:
        status = BlurStatus.BLURRY

    return BlurMetrics(score=score, status=status)


def assess_skew(image: np.ndarray, config: ImageQualityConfig = IMAGE_QUALITY_CONFIG) -> SkewMetrics:
    """
    Estimate page rotation using the minimum-area-bounding-rectangle method.

    Approach: threshold the image to isolate dark (ink) pixels, find the
    coordinates of all foreground pixels, and fit the minimum-area
    rotated rectangle around them with ``cv2.minAreaRect``. The angle is
    then derived directly from the geometry of that rectangle's corner
    points (via ``cv2.boxPoints``) rather than from OpenCV's internal
    angle field: that field's numeric convention has changed across
    OpenCV versions (pre-4.5 vs. 4.5+ report angles on different ranges,
    and near-square content can flip which side is treated as "width"),
    which made angle sign/range interpretation version-dependent. Reading
    the angle straight from the rectangle's edge vectors sidesteps that
    entirely and gives the same result on any OpenCV version.

    For a page of roughly horizontal text/content, the edge closest to
    horizontal approximates the page's skew angle. This works best when
    there is a reasonable amount of foreground content (mostly-blank
    pages yield an unreliable/near-zero estimate, which we treat
    conservatively as "no skew detected" rather than guessing).

    Args:
        image: Grayscale or RGB image array.
        config: Threshold (in degrees) above which deskewing is recommended.

    Returns:
        A :class:`SkewMetrics` instance.
    """
    gray = _validate_image(image)

    # Foreground = dark pixels on a lighter background, typical of scanned text.
    _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(thresholded)

    if coords is None or len(coords) < 20:
        # Not enough foreground content to estimate an orientation reliably.
        logger.debug("Insufficient foreground pixels for skew estimation; defaulting to 0.0 degrees.")
        angle = 0.0
    else:
        rect = cv2.minAreaRect(coords)
        box_points = cv2.boxPoints(rect)

        # Compute the orientation of each of the rectangle's 4 edges
        # relative to horizontal, normalized to (-90, 90], and take the
        # edge closest to horizontal as the estimated skew angle. This is
        # equivalent for opposite edges (a rectangle has 2 edge
        # directions), so effectively we're picking whichever of the two
        # edge directions is more horizontal.
        edge_angles = []
        for i in range(4):
            p1, p2 = box_points[i], box_points[(i + 1) % 4]
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            edge_angle = float(np.degrees(np.arctan2(dy, dx)))
            edge_angle = ((edge_angle + 90) % 180) - 90  # normalize to (-90, 90]
            edge_angles.append(edge_angle)

        angle = min(edge_angles, key=abs)

    status = (
        SkewStatus.NEEDS_DESKEW
        if abs(angle) > config.skew_threshold_degrees
        else SkewStatus.ACCEPTABLE
    )

    return SkewMetrics(angle_degrees=float(angle), status=status)


def _order_corners(points: np.ndarray) -> List[Tuple[float, float]]:
    """
    Order 4 arbitrary points as (top-left, top-right, bottom-right, bottom-left).

    Standard trick: top-left has the smallest (x+y) sum and bottom-right
    the largest; top-right has the smallest (y-x) difference and
    bottom-left the largest. Works regardless of the order
    ``cv2.approxPolyDP`` happened to return the points in.
    """
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).flatten()

    top_left = points[np.argmin(sums)]
    bottom_right = points[np.argmax(sums)]
    top_right = points[np.argmin(diffs)]
    bottom_left = points[np.argmax(diffs)]

    return [
        (float(top_left[0]), float(top_left[1])),
        (float(top_right[0]), float(top_right[1])),
        (float(bottom_right[0]), float(bottom_right[1])),
        (float(bottom_left[0]), float(bottom_left[1])),
    ]


def _perspective_candidate_score(
    corners: np.ndarray,
    frame_width: int,
    frame_height: int,
    area_ratio: float,
    config: ImageQualityConfig,
) -> float:
    """Score a quadrilateral as a likely outer document boundary.

    Area alone is deliberately not enough: photographed pages often contain
    strong inner rectangles (tables, boxes, logos), while dilation can also
    merge the image border into a large false contour.  We therefore prefer
    large, convex, roughly rectangular candidates that lie close to the
    frame boundary, while rejecting the characteristic full-frame merged
    contour artifact.
    """
    points = corners.astype(np.float64)
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)

    # A contour that touches two opposite frame sides but occupies less than
    # most of the frame is usually a border/background merge.  Keep genuinely
    # full-frame pages eligible.
    eps_x = config.perspective_edge_touch_epsilon * frame_width
    eps_y = config.perspective_edge_touch_epsilon * frame_height
    touches_horizontal_pair = x_min <= eps_x and x_max >= frame_width - eps_x
    touches_vertical_pair = y_min <= eps_y and y_max >= frame_height - eps_y
    if (
        (touches_horizontal_pair or touches_vertical_pair)
        and area_ratio < config.perspective_border_artifact_area_ratio
    ):
        return -1.0

    # Larger pages are preferred, but area is capped so it cannot dominate
    # the score over the much more useful boundary-location signal.
    area_score = min(area_ratio / 0.65, 1.0)

    margins = np.array(
        [
            x_min / frame_width,
            (frame_width - x_max) / frame_width,
            y_min / frame_height,
            (frame_height - y_max) / frame_height,
        ],
        dtype=np.float64,
    )
    mean_margin = float(np.clip(np.mean(margins), 0.0, 1.0))
    margin_scale = max(config.perspective_border_margin_scale, 1e-6)
    border_score = 1.0 - min(mean_margin / margin_scale, 1.0)

    # Prefer document-like aspect ratios without hard-coding portrait A4.
    side_lengths = np.array(
        [
            np.linalg.norm(points[1] - points[0]),
            np.linalg.norm(points[2] - points[1]),
            np.linalg.norm(points[3] - points[2]),
            np.linalg.norm(points[0] - points[3]),
        ],
        dtype=np.float64,
    )
    short_side = max(float(min(side_lengths)), 1e-6)
    long_side = float(max(side_lengths))
    aspect_ratio = long_side / short_side
    aspect_score = 1.0 - min(
        abs(math.log(max(aspect_ratio, 1e-6) / math.sqrt(2.0))) / math.log(2.0),
        1.0,
    )

    # Good document corners tend to be close to right angles.
    corner_scores = []
    for i in range(4):
        prev_point = points[(i - 1) % 4] - points[i]
        next_point = points[(i + 1) % 4] - points[i]
        denom = np.linalg.norm(prev_point) * np.linalg.norm(next_point)
        if denom <= 1e-9:
            corner_scores.append(0.0)
            continue
        cosine = abs(float(np.dot(prev_point, next_point) / denom))
        corner_scores.append(1.0 - min(cosine, 1.0))
    corner_score = float(np.mean(corner_scores))

    # Smoothly penalize candidates deep inside the frame. This prevents an
    # inner table from winning simply because it forms a very clean rectangle.
    interior_penalty = 0.45 * min(mean_margin / margin_scale, 1.0)

    return (
        0.30 * area_score
        + 0.35 * border_score
        + 0.15 * aspect_score
        + 0.10 * 1.0  # candidates are filtered for convexity before scoring
        + 0.10 * corner_score
        - interior_penalty
    )


def assess_perspective(
    image: np.ndarray, config: ImageQualityConfig = IMAGE_QUALITY_CONFIG
) -> PerspectiveMetrics:
    """
    Detect the outer document boundary in a photographed page.

    Multiple four-sided contour candidates are considered instead of simply
    taking the largest contour.  The candidate score combines coverage,
    proximity to the image border, document-like geometry, and corner
    quality.  This is important for real photographs where an inner table or
    box can be a cleaner/larger contour than a partially occluded page edge.

    The detector also rejects a characteristic false positive produced when
    dilation merges the image border into one edge-to-edge contour: if that
    contour touches opposite frame sides but covers less than the configured
    "full-frame" ratio, it is treated as an artifact rather than as the page.

    The returned corners are ordered top-left, top-right, bottom-right,
    bottom-left and refer to the original image coordinate system.
    """
    gray = _validate_image(image)
    frame_height, frame_width = gray.shape[:2]
    frame_area = float(frame_height * frame_width)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(
        blurred,
        config.perspective_canny_low,
        config.perspective_canny_high,
    )
    if config.perspective_dilate_iterations > 0:
        edges = cv2.dilate(
            edges,
            np.ones((5, 5), np.uint8),
            iterations=config.perspective_dilate_iterations,
        )

    contours, _ = cv2.findContours(
        edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    candidates: List[Tuple[float, float, np.ndarray, int]] = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue

        seen_shapes = set()
        for epsilon_index, epsilon in enumerate(config.perspective_approx_epsilons):
            approx = cv2.approxPolyDP(contour, epsilon * perimeter, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue

            quad = approx.reshape(4, 2).astype(np.float64)
            key = tuple(np.round(quad.ravel(), 1))
            if key in seen_shapes:
                continue
            seen_shapes.add(key)

            area = abs(float(cv2.contourArea(approx)))
            coverage_ratio = area / frame_area if frame_area > 0 else 0.0
            if coverage_ratio < config.perspective_min_coverage_ratio:
                continue

            score = _perspective_candidate_score(
                quad,
                frame_width,
                frame_height,
                coverage_ratio,
                config,
            )
            if score < 0:
                continue

            # Prefer a tighter polygon approximation when candidates are
            # otherwise close. Larger epsilon values can move corners away
            # from the true document edge while making a contour look
            # deceptively clean.
            score -= 0.05 * epsilon_index
            candidates.append((score, coverage_ratio, quad, epsilon_index))

    if not candidates:
        logger.debug(
            "No plausible document quadrilateral found (minimum coverage %.0f%%).",
            config.perspective_min_coverage_ratio * 100,
        )
        return PerspectiveMetrics(
            corners=None,
            frame_coverage_ratio=0.0,
            status=PerspectiveStatus.UNDETECTED,
        )

    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, coverage_ratio, best_quad, _ = candidates[0]

    if best_score < config.perspective_min_confidence:
        logger.debug(
            "Best document quadrilateral confidence %.2f is below %.2f; "
            "skipping perspective correction.",
            best_score,
            config.perspective_min_confidence,
        )
        return PerspectiveMetrics(
            corners=None,
            frame_coverage_ratio=coverage_ratio,
            status=PerspectiveStatus.UNDETECTED,
        )

    corners = _order_corners(best_quad)

    if coverage_ratio >= config.perspective_max_coverage_ratio:
        status = PerspectiveStatus.NOT_NEEDED
    else:
        status = PerspectiveStatus.NEEDS_CORRECTION

    logger.debug(
        "Selected document quadrilateral: confidence=%.2f, coverage=%.1f%%, corners=%s.",
        best_score,
        coverage_ratio * 100,
        [(round(x, 1), round(y, 1)) for x, y in corners],
    )

    return PerspectiveMetrics(
        corners=corners,
        frame_coverage_ratio=coverage_ratio,
        status=status,
    )

def assess_contrast(image: np.ndarray, config: ImageQualityConfig = IMAGE_QUALITY_CONFIG) -> ContrastMetrics:
    """
    Measure global contrast as the standard deviation of grayscale intensities.

    Args:
        image: Grayscale or RGB image array.
        config: Thresholds separating GOOD / MODERATE / LOW contrast.

    Returns:
        A :class:`ContrastMetrics` instance.
    """
    gray = _validate_image(image)
    score = float(gray.std())

    if score >= config.contrast_good_threshold:
        status = ContrastStatus.GOOD
    elif score >= config.contrast_low_threshold:
        status = ContrastStatus.MODERATE
    else:
        status = ContrastStatus.LOW

    return ContrastMetrics(score=score, status=status)


def assess_noise(image: np.ndarray, config: ImageQualityConfig = IMAGE_QUALITY_CONFIG) -> NoiseMetrics:
    """
    Estimate noise level using the Immerkaer (1996) fast noise estimator.

    See the :class:`NoiseMetrics` docstring for the method's rationale
    and explicit limitations. This is intentionally a conservative,
    classical estimate rather than a fabricated precise-looking score.

    Args:
        image: Grayscale or RGB image array.
        config: Thresholds separating LOW / MODERATE / HIGH noise.

    Returns:
        A :class:`NoiseMetrics` instance.
    """
    gray = _validate_image(image).astype(np.float64)
    height, width = gray.shape[:2]

    if height < 3 or width < 3:
        # Estimator requires interior pixels; be conservative and report LOW
        # rather than fabricate a score from too little data.
        logger.debug("Image too small for reliable noise estimation; reporting LOW/0.0.")
        return NoiseMetrics(score=0.0, status=NoiseStatus.LOW)

    # Immerkaer's noise-estimation kernel (Laplacian-of-Gaussian-like, but
    # calibrated so its response to noise is separable from response to edges).
    kernel = np.array(
        [[1, -2, 1], [-2, 4, -2], [1, -2, 1]],
        dtype=np.float64,
    )
    convolved = cv2.filter2D(gray, ddepth=cv2.CV_64F, kernel=kernel)
    sigma = np.sqrt(np.pi / 2.0) / (6.0 * (width - 2) * (height - 2)) * np.sum(np.abs(convolved))
    score = float(sigma)

    if score >= config.noise_high_threshold:
        status = NoiseStatus.HIGH
    elif score >= config.noise_moderate_threshold:
        status = NoiseStatus.MODERATE
    else:
        status = NoiseStatus.LOW

    return NoiseMetrics(score=score, status=status)


# --------------------------------------------------------------------------
# Overall quality + recommendations
# --------------------------------------------------------------------------

# Per-metric severity used to derive an overall rating. This is a simple,
# transparent additive rule (not an unexplained weighted formula): each
# metric contributes 0 (fine), 1 (mild issue), or 2 (significant issue),
# and the total maps to a quality band. It's intentionally easy to read
# and to retune once real OCR-accuracy data is available.
_SEVERITY_BANDS = {
    0: OverallQuality.EXCELLENT,
    1: OverallQuality.GOOD,
}


def _overall_quality_from_severity(total_severity: int) -> OverallQuality:
    """Map a total severity score (0-10) to an overall quality band."""
    if total_severity == 0:
        return OverallQuality.EXCELLENT
    if total_severity <= 1:
        return OverallQuality.GOOD
    if total_severity <= 3:
        return OverallQuality.FAIR
    return OverallQuality.POOR


def _build_recommendations(
    resolution: ResolutionMetrics,
    blur: BlurMetrics,
    skew: SkewMetrics,
    contrast: ContrastMetrics,
    noise: NoiseMetrics,
    perspective: PerspectiveMetrics,
    config: ImageQualityConfig,
) -> List[PreprocessingRecommendation]:
    """Derive explainable preprocessing recommendations from measurements."""
    recommendations: List[PreprocessingRecommendation] = []

    if perspective.status is PerspectiveStatus.NEEDS_CORRECTION:
        reason = (
            f"Detected a document boundary covering {perspective.frame_coverage_ratio:.0%} "
            f"of the frame; perspective correction can crop to just the page and "
            f"straighten it before other preprocessing runs."
        )
        recommendations.append(
            PreprocessingRecommendation(PreprocessingOperation.PERSPECTIVE_CORRECTION, reason)
        )

    if resolution.is_low_resolution:
        if resolution.dpi is not None:
            reason = (
                f"Detected resolution is {resolution.dpi} DPI, below the "
                f"configured minimum of {config.min_dpi} DPI."
            )
        else:
            reason = (
                f"DPI unavailable; pixel dimensions {resolution.width}x{resolution.height} "
                f"fall below the configured minimum of "
                f"{config.min_width_px}x{config.min_height_px}."
            )
        recommendations.append(PreprocessingRecommendation(PreprocessingOperation.RESIZE, reason))

    if blur.status in (BlurStatus.BLURRY, BlurStatus.MODERATELY_BLURRY):
        reason = (
            f"Blur score {blur.score:.1f} is below the sharp threshold of "
            f"{config.blur_sharp_threshold:.1f} (status: {blur.status.value})."
        )
        recommendations.append(PreprocessingRecommendation(PreprocessingOperation.DENOISE, reason))

    if skew.status is SkewStatus.NEEDS_DESKEW:
        reason = (
            f"Estimated skew angle is {skew.angle_degrees:.1f}°, above the "
            f"configured threshold of {config.skew_threshold_degrees:.1f}°."
        )
        recommendations.append(PreprocessingRecommendation(PreprocessingOperation.DESKEW, reason))

    if contrast.status is ContrastStatus.LOW:
        reason = (
            f"Contrast score {contrast.score:.1f} is below the configured "
            f"low-contrast threshold of {config.contrast_low_threshold:.1f}."
        )
        recommendations.append(
            PreprocessingRecommendation(PreprocessingOperation.CONTRAST_ENHANCEMENT, reason)
        )
    elif contrast.status is ContrastStatus.MODERATE:
        reason = (
            f"Contrast score {contrast.score:.1f} is moderate (below "
            f"{config.contrast_good_threshold:.1f}); thresholding may help binarization."
        )
        recommendations.append(PreprocessingRecommendation(PreprocessingOperation.THRESHOLDING, reason))

    if noise.status is NoiseStatus.HIGH:
        reason = (
            f"Estimated noise level {noise.score:.2f} exceeds the configured "
            f"high-noise threshold of {config.noise_high_threshold:.2f}."
        )
        # Avoid recommending DENOISE twice if blur already triggered it.
        if not any(r.operation is PreprocessingOperation.DENOISE for r in recommendations):
            recommendations.append(PreprocessingRecommendation(PreprocessingOperation.DENOISE, reason))

    return recommendations


def assess_page_quality(
    image: np.ndarray,
    page_number: int,
    dpi: Optional[int] = None,
    config: ImageQualityConfig = IMAGE_QUALITY_CONFIG,
) -> PageQualityReport:
    """
    Run the full Phase 2 quality assessment on a single rendered page.

    This is the main entry point for the module and is designed to
    process exactly one page independently of any others, so it can
    later be dropped into a batch/queue runner without modification.

    Args:
        image: Rendered page image, as an RGB or grayscale numpy array
            (e.g. from :func:`src.pdf.page_renderer.render_page_to_array`).
        page_number: 1-indexed page number, for reporting purposes.
        dpi: Known rendering DPI, if available.
        config: Quality thresholds to use for this assessment.

    Returns:
        A :class:`PageQualityReport`.

    Raises:
        ImageQualityError: If the image cannot be analyzed at all
            (missing, empty, wrong shape, etc.).
    """
    _validate_image(image)  # fail fast with a clear error before doing per-metric work

    resolution = assess_resolution(image, dpi=dpi, config=config)
    blur = assess_blur(image, config=config)
    skew = assess_skew(image, config=config)
    contrast = assess_contrast(image, config=config)
    noise = assess_noise(image, config=config)
    perspective = assess_perspective(image, config=config)

    severity = (
        int(resolution.is_low_resolution)
        + {BlurStatus.SHARP: 0, BlurStatus.MODERATELY_BLURRY: 1, BlurStatus.BLURRY: 2}[blur.status]
        + (1 if skew.status is SkewStatus.NEEDS_DESKEW else 0)
        + {ContrastStatus.GOOD: 0, ContrastStatus.MODERATE: 1, ContrastStatus.LOW: 2}[contrast.status]
        + {NoiseStatus.LOW: 0, NoiseStatus.MODERATE: 1, NoiseStatus.HIGH: 2}[noise.status]
        + (1 if perspective.status is PerspectiveStatus.NEEDS_CORRECTION else 0)
    )
    overall_quality = _overall_quality_from_severity(severity)

    recommendations = _build_recommendations(resolution, blur, skew, contrast, noise, perspective, config)

    logger.info(
        "Page %d quality: %s (severity=%d, recommendations=%s)",
        page_number,
        overall_quality.value,
        severity,
        [r.operation.value for r in recommendations] or ["NONE"],
    )

    return PageQualityReport(
        page_number=page_number,
        resolution=resolution,
        blur=blur,
        skew=skew,
        contrast=contrast,
        noise=noise,
        perspective=perspective,
        overall_quality=overall_quality,
        recommendations=recommendations,
    )


def assess_page_quality_safe(
    image: Optional[np.ndarray],
    page_number: int,
    dpi: Optional[int] = None,
    config: ImageQualityConfig = IMAGE_QUALITY_CONFIG,
) -> Optional[PageQualityReport]:
    """
    Same as :func:`assess_page_quality`, but never raises.

    Intended for batch/CLI contexts where one bad page (corrupt render,
    unreadable image, unexpected format) must not stop processing of
    the rest of the document. Logs the error and returns ``None``.
    """
    try:
        return assess_page_quality(image, page_number=page_number, dpi=dpi, config=config)
    except ImageQualityError as exc:
        logger.error("Quality assessment failed for page %d: %s", page_number, exc)
        return None
    except Exception as exc:  # defensive: OpenCV can raise its own exception types
        logger.error("Unexpected error assessing page %d: %s", page_number, exc)
        return None