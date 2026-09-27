"""
Centralized, environment-independent configuration for Phase 1.

All paths are resolved relative to the project root so the codebase
never relies on hardcoded absolute paths and works the same way
regardless of where it is checked out or executed from.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2


# Project root = two levels up from this file (src/utils/config.py -> project root)
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"
PROCESSED_DIR: Path = DATA_DIR / "processed"
RENDERED_IMAGES_DIR: Path = PROCESSED_DIR / "images"


@dataclass(frozen=True)
class PDFTypeDetectionConfig:
    """Configurable thresholds for PDF type detection."""

    # Minimum number of non-whitespace characters on a page for it to be
    # considered "text-based" rather than "scanned/image-based".
    min_text_chars_per_page: int = 50


@dataclass(frozen=True)
class PageRenderConfig:
    """Configurable defaults for page rendering."""

    default_dpi: int = 200
    output_dir: Path = RENDERED_IMAGES_DIR
    image_format: str = "png"


@dataclass(frozen=True)
class ImageQualityConfig:
    """
    Configurable thresholds for Phase 2 image-quality assessment.

    IMPORTANT: These are reasonable *starting* values based on common
    OCR-preprocessing rules of thumb, not values tuned on real company
    documents. They should be recalibrated once ground-truth OCR
    accuracy data is available for your actual scanner/document mix.
    """

    # --- Resolution ---
    # Below this DPI (when DPI is known) a page is considered low resolution.
    min_dpi: int = 200
    # Fallback pixel-dimension floor used when DPI cannot be determined.
    min_width_px: int = 1000
    min_height_px: int = 1000

    # --- Blur (variance of Laplacian) ---
    # Below this variance, a page is BLURRY. Above the sharp threshold, SHARP.
    # Between the two, MODERATELY_BLURRY. Sensitive to render DPI/content
    # density, so recalibrate per scanning pipeline.
    blur_sharp_threshold: float = 150.0
    blur_blurry_threshold: float = 50.0

    # --- Skew / rotation ---
    # Estimated skew angle (degrees) above which deskewing is recommended.
    skew_threshold_degrees: float = 2.0

    # --- Contrast (std deviation of grayscale intensities, 0-255 scale) ---
    contrast_good_threshold: float = 50.0
    contrast_low_threshold: float = 25.0

    # --- Noise (Immerkaer fast noise estimator, approx. sigma) ---
    noise_moderate_threshold: float = 2.0
    noise_high_threshold: float = 6.0

    # --- Perspective (largest well-scored 4-sided contour vs. frame area) ---
    # Candidates below this coverage aren't worth scoring at all (stray
    # edges, not a plausible page).
    perspective_min_coverage_ratio: float = 0.15
    # A detected quadrilateral covering at least this fraction of the
    # frame is treated as "the page already fills the frame" — nothing
    # meaningful to crop away or straighten.
    perspective_max_coverage_ratio: float = 0.97
    # Canny edge-detection thresholds. Deliberately loose (the default
    # 50/150 "textbook" values miss the weaker page-vs-background edge a
    # real phone photo often has — e.g. where a hand partially shadows
    # the boundary) — but loose thresholds pick up more background
    # texture too, which is exactly why the scoring below (not just
    # raw contour area) is what actually picks the right candidate.
    perspective_canny_low: int = 20
    perspective_canny_high: int = 60
    # Dilation iterations before contour-finding, to close small gaps in
    # the detected page edge (e.g. where an occluding hand breaks it).
    perspective_dilate_iterations: int = 2
    # Polygon-approximation tolerances (as a fraction of contour
    # perimeter) tried in order until one simplifies a contour to a
    # clean 4-point convex shape. A single fixed tolerance (the original
    # implementation used 0.02 only) works for a clean synthetic
    # rectangle but is often too tight for a real photographed page's
    # slightly irregular outline.
    perspective_approx_epsilons: Tuple[float, ...] = (0.01, 0.02, 0.03, 0.05, 0.08)
    # A composite candidate score (area + border proximity + corner
    # quality + aspect ratio, each 0-1) must reach this to be trusted at
    # all — this is the "confidence threshold" gate, separate from the
    # coverage-ratio floor above, which only filters obviously-too-small
    # candidates before scoring.
    perspective_min_confidence: float = 0.55
    # Margin (as a fraction of frame width/height) beyond which the
    # border-proximity score bottoms out at 0. A candidate centered deep
    # in the frame scores 0 here; one hugging the frame edges scores
    # near 1 — but per requirement, this is a smooth preference, not a
    # requirement that a real document must touch the frame boundary.
    perspective_border_margin_scale: float = 0.5
    # A candidate whose bounding box spans a full frame dimension
    # (top-and-bottom or left-and-right) edge-to-edge, within this
    # margin fraction, but *doesn't* actually cover most of the frame,
    # is excluded outright: that combination is the signature of a
    # contour-merging artifact (edges fused with the image's own border
    # or an unrelated background line by dilation), not a real page.
    perspective_edge_touch_epsilon: float = 0.01
    perspective_border_artifact_area_ratio: float = 0.85


@dataclass(frozen=True)
class PreprocessingConfig:
    """
    Configurable parameters for Phase 3 image preprocessing.

    Where a Phase 3 decision should match a Phase 2 threshold (e.g. "how
    much resolution is enough"), that threshold is read directly from
    :class:`ImageQualityConfig` instead of being duplicated here — this
    class only holds settings that are specific to *performing* an
    operation, not to *deciding whether* one is needed (that remains
    Phase 2's job).
    """

    # --- Resize ---
    # Upscaling is capped so a tiny/degenerate image (e.g. a thumbnail)
    # can't be blown up into an unreasonably large, mostly-interpolated
    # image. Interpolation uses a smooth kernel appropriate for enlarging
    # photographic/scanned content without obvious blockiness.
    max_upscale_factor: float = 4.0
    resize_interpolation: int = cv2.INTER_CUBIC

    # --- Denoise (OpenCV fastNlMeansDenoising[Colored]) ---
    # `h` / `hColor` control filter strength: higher removes more noise
    # but can also soften fine detail (thin character strokes), so these
    # are kept moderate rather than at OpenCV's more aggressive defaults.
    denoise_h_luminance: float = 7.0
    denoise_h_color: float = 7.0
    denoise_template_window_size: int = 7
    denoise_search_window_size: int = 21

    # --- Deskew ---
    # Below this angle, rotating would introduce interpolation blur for
    # a correction too small to be visually or OCR-meaningful.
    deskew_min_angle_degrees: float = 0.1
    deskew_interpolation: int = cv2.INTER_CUBIC
    # Fill color for corners exposed by rotation (white, matching a
    # typical document background) so no artificial dark borders appear.
    deskew_border_value: int = 255

    # --- Contrast (CLAHE) ---
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8

    # --- Thresholding / binarization ---
    # "otsu": single global threshold chosen automatically from the
    #   image histogram — simple, deterministic, and consistent with
    #   the OTSU method already used by Phase 2's assess_skew().
    # "adaptive": a locally-computed threshold per neighborhood, better
    #   suited to pages with uneven illumination.
    threshold_method: str = "otsu"
    adaptive_block_size: int = 35  # must be odd; enforced at call time
    adaptive_c: int = 15

    # --- Perspective correction ---
    # Interpolation for the perspective warp itself.
    perspective_interpolation: int = cv2.INTER_CUBIC


@dataclass(frozen=True)
class OCRConfig:
    """
    Configurable parameters for Phase 4 OCR.

    Engine-agnostic at the top (engine selection, language, confidence
    handling); the Unlimited-OCR-specific knobs (model name/device/
    dtype/generation length) are grouped separately since they're
    meaningful only to that one engine.
    """

    # Which OCR engine implementation to use — selected by name through
    # src/ocr/factory.py so the rest of the pipeline never hardcodes a
    # specific engine class. Baidu Unlimited-OCR ("unlimited") is the
    # project's only supported engine; any other value is a controlled
    # OCREngineNotAvailableError, not a silent fallback. Overridable
    # via the OCR_ENGINE environment variable without touching code.
    engine: str = os.environ.get("OCR_ENGINE", "unlimited")

    # Trained language data the engine should use. Project documents
    # are a mix of Arabic and English, so the default requests both
    # rather than privileging either one; callers needing a single
    # language (e.g. "eng" or "ara" alone) pass it explicitly.
    # Unlimited-OCR is a vision-language model without discrete
    # per-language packs — this is currently informational/carried
    # through to the result rather than changing model behavior.
    language: str = "ara+eng"

    # Recognized words/regions are reported with a confidence on a
    # 0-100 scale; structural (non-word) rows are reported as -1. Only
    # rows at/above this threshold are kept in the result, which
    # excludes those -1 rows by default without needing a special
    # case. Unlimited-OCR does not report real confidence (every row
    # is a documented 0.0 placeholder — see src/ocr/unlimited_ocr.py),
    # so with the default threshold every region is still kept; only a
    # positive threshold would start excluding them.
    min_confidence: float = 0.0

    # --- Confidence bucketing (see src.ocr.models.confidence_level) ---
    # A page's mean confidence is not an absolute measure of OCR
    # correctness — only an indicator of how sure the engine was. For
    # Unlimited-OCR specifically, check
    # PageOCRResult.metadata["confidence_available"] before treating
    # this bucket as meaningful at all (see src/ocr/ocr.py).
    confidence_very_good_threshold: float = 90.0
    confidence_good_threshold: float = 75.0
    confidence_moderate_threshold: float = 50.0

    # --- Unlimited-OCR-specific ---
    # Hugging Face model id. Overridable via the UNLIMITED_OCR_MODEL
    # environment variable.
    unlimited_model_name: str = os.environ.get("UNLIMITED_OCR_MODEL", "baidu/Unlimited-OCR")

    # Torch device string (e.g. "cuda", "cuda:0", "cpu"). Unlimited-OCR
    # is a large multimodal model; GPU execution is the supported/
    # expected path, not CPU. Overridable via UNLIMITED_OCR_DEVICE.
    unlimited_device: str = os.environ.get("UNLIMITED_OCR_DEVICE", "cuda")

    # Torch dtype name the model weights are loaded in. Overridable
    # via UNLIMITED_OCR_DTYPE.
    unlimited_dtype: str = os.environ.get("UNLIMITED_OCR_DTYPE", "bfloat16")

    # Generation length cap passed to the model per page. Overridable
    # via UNLIMITED_OCR_MAX_NEW_TOKENS.
    unlimited_max_new_tokens: int = int(os.environ.get("UNLIMITED_OCR_MAX_NEW_TOKENS", "8192"))


# Single shared instances used across the app unless overridden explicitly.
PDF_TYPE_DETECTION_CONFIG = PDFTypeDetectionConfig()
PAGE_RENDER_CONFIG = PageRenderConfig()
IMAGE_QUALITY_CONFIG = ImageQualityConfig()
PREPROCESSING_CONFIG = PreprocessingConfig()
OCR_CONFIG = OCRConfig()


def ensure_directories() -> None:
    """Create data directories used by the pipeline if they don't exist."""
    RENDERED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)