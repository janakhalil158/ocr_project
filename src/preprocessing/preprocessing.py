"""
Phase 3: Document image preprocessing.

Phase 2 (:mod:`src.quality.image_quality`) MEASURES a page and produces
a :class:`~src.quality.image_quality.PageQualityReport` that RECOMMENDS
which preprocessing operations would help, without performing any of
them. Phase 3 performs those recommended operations:

    Input page image
           |
           v
    Phase 2: assess_page_quality()
           |
           v
    PageQualityReport
           |
           v
    Phase 3: preprocess_page()
           |
           v
    ProcessedPage (image + operations_applied)
           |
           v
    Future OCR phase

Each operation (:func:`resize_image`, :func:`correct_perspective`,
:func:`denoise_image`, :func:`deskew_image`, :func:`enhance_contrast`,
:func:`threshold_image`) is a small, independently testable function
that performs exactly one transform and knows nothing about
recommendations or reports. The high-level :func:`preprocess_page` is
the only place that reads a :class:`PageQualityReport` and decides
which of those functions to call.

:func:`correct_perspective` is a little different from the others: its
"is this needed" decision (does a document-sized quadrilateral exist
in the frame at all?) is expensive enough that it isn't a cheap scalar
threshold check the way blur/skew/contrast are — so that detection
work happens once, up front, in Phase 2's
:func:`~src.quality.image_quality.assess_perspective`, and this module
only performs the resulting warp, exactly mirroring how
:func:`assess_skew` measures an angle that :func:`deskew_image` then
acts on.

Design goals (mirroring Phase 2):
    * Classical, explainable OpenCV techniques only — no deep learning,
      appropriate for a first pass over large document volumes.
    * Operations are skipped rather than blindly applied whenever they
      can tell, from the input itself, that they wouldn't help (e.g.
      resizing an already-adequate image, deskewing an already-straight
      one) — even if called directly rather than through the pipeline.
    * Preserve image integrity: dimensions, aspect ratio, and channel
      layout are kept unless an operation inherently changes them
      (thresholding necessarily produces a single-channel binary image).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.quality.image_quality import PageQualityReport, PreprocessingOperation
from src.utils.config import IMAGE_QUALITY_CONFIG, PREPROCESSING_CONFIG, PreprocessingConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PreprocessingError(Exception):
    """Raised when a page image cannot be preprocessed at all (invalid input)."""


# --------------------------------------------------------------------------
# Result object
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ProcessedPage:
    """Result of running Phase 3 preprocessing on a single page."""

    page_number: int
    image: np.ndarray
    operations_applied: List[PreprocessingOperation] = field(default_factory=list)

    @property
    def operation_names(self) -> List[str]:
        """Flat list of applied operation names, e.g. for JSON output."""
        return [op.value for op in self.operations_applied]

    def to_dict(self) -> dict:
        """Plain-dict representation (excludes the raw image array)."""
        height, width = self.image.shape[:2]
        channels = 1 if self.image.ndim == 2 else self.image.shape[2]
        return {
            "page_number": self.page_number,
            "width": width,
            "height": height,
            "channels": channels,
            "operations_applied": self.operation_names,
        }


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _validate_image(image: np.ndarray) -> np.ndarray:
    """Validate that an array looks like a usable, processable image."""
    if image is None:
        raise PreprocessingError("Received a None image.")
    if not isinstance(image, np.ndarray):
        raise PreprocessingError(f"Expected a numpy.ndarray, got {type(image)}.")
    if image.size == 0 or image.ndim not in (2, 3):
        raise PreprocessingError(f"Image has an invalid shape: {getattr(image, 'shape', None)}")
    if image.ndim == 3 and image.shape[2] not in (3, 4):
        raise PreprocessingError(f"Unsupported number of channels: {image.shape}")
    if image.dtype != np.uint8:
        raise PreprocessingError(f"Expected a uint8 image, got dtype {image.dtype}.")

    height, width = image.shape[:2]
    if height < 3 or width < 3:
        raise PreprocessingError(f"Image is too small to process: {image.shape}")

    return image


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an RGB(A) or grayscale image array to single-channel grayscale."""
    if image.ndim == 2:
        return image
    code = cv2.COLOR_RGB2GRAY if image.shape[2] == 3 else cv2.COLOR_RGBA2GRAY
    return cv2.cvtColor(image, code)


# --------------------------------------------------------------------------
# Individual operations
# --------------------------------------------------------------------------


def resize_image(
    image: np.ndarray,
    current_dpi: Optional[int] = None,
    config: PreprocessingConfig = PREPROCESSING_CONFIG,
) -> np.ndarray:
    """
    Upscale a low-resolution page to a usable size, preserving aspect ratio.

    Adequacy is judged against the same thresholds Phase 2 uses to flag
    low resolution (:class:`ImageQualityConfig`'s ``min_dpi`` when a DPI
    is known, otherwise ``min_width_px`` / ``min_height_px``), so Phase 3
    resizes exactly the pages Phase 2 would flag, and no others. An
    already-adequate image is returned unchanged (same array, no copy).

    Args:
        image: Grayscale or RGB(A) uint8 image array.
        current_dpi: Known rendering DPI of ``image``, if available.
        config: Preprocessing settings (interpolation, max upscale factor).

    Returns:
        The resized image, or the original array unchanged if it is
        already adequate.
    """
    _validate_image(image)
    height, width = image.shape[:2]
    quality_config = IMAGE_QUALITY_CONFIG

    if current_dpi is not None:
        if current_dpi >= quality_config.min_dpi:
            return image
        scale = quality_config.min_dpi / current_dpi
    else:
        scale = max(
            quality_config.min_width_px / width,
            quality_config.min_height_px / height,
        )
        if scale <= 1.0:
            return image

    if scale > config.max_upscale_factor:
        logger.warning(
            "Required upscale factor %.2fx exceeds the configured cap of "
            "%.2fx; capping to avoid an excessively enlarged image.",
            scale,
            config.max_upscale_factor,
        )
        scale = config.max_upscale_factor

    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))

    logger.info(
        "Resizing page from %dx%d to %dx%d (scale=%.2fx)",
        width,
        height,
        new_width,
        new_height,
        scale,
    )
    return cv2.resize(image, (new_width, new_height), interpolation=config.resize_interpolation)


# --------------------------------------------------------------------------
# Perspective correction
# --------------------------------------------------------------------------


def correct_perspective(
    image: np.ndarray,
    corners: Optional[List[Tuple[float, float]]],
    config: PreprocessingConfig = PREPROCESSING_CONFIG,
) -> np.ndarray:
    """
    Warp a photographed, trapezoidal document into a straight rectangle.

    Given the four page corners
    :func:`~src.quality.image_quality.assess_perspective` detected
    (ordered top-left, top-right, bottom-right, bottom-left), this
    computes a destination rectangle sized from the corners' own edge
    lengths — so the output isn't stretched to an arbitrary aspect
    ratio — then applies a perspective transform via
    ``cv2.getPerspectiveTransform`` / ``cv2.warpPerspective``. This both
    straightens the page (handling genuine trapezoidal distortion from
    an angled photo, not just simple in-plane rotation) and crops away
    surrounding background in the same step.

    Matching this module's general design (see the module docstring):
    if there's nothing usable to act on, this returns the original
    image unchanged rather than raising — ``corners=None`` (Phase 2
    found no reliable document boundary) and a degenerate/too-small
    detected region are both treated this way, so it's safe to call
    directly with whatever Phase 2 produced, without a separate
    "did detection succeed" check at every call site.

    Args:
        image: Grayscale or RGB(A) uint8 image array.
        corners: Exactly 4 (x, y) points, ordered (top-left, top-right,
            bottom-right, bottom-left), or ``None`` if no document
            boundary was reliably detected.
        config: Preprocessing settings (warp interpolation).

    Returns:
        The perspective-corrected image, cropped to the detected page
        boundary — or the original ``image``, unchanged, if ``corners``
        is ``None`` or produces a degenerate output.
    """
    _validate_image(image)

    if corners is None:
        logger.debug("No document corners provided; skipping perspective correction.")
        return image
    if len(corners) != 4:
        raise PreprocessingError(f"Expected exactly 4 corners, got {len(corners)}.")

    top_left, top_right, bottom_right, bottom_left = (np.array(p, dtype=np.float32) for p in corners)

    width_top = float(np.linalg.norm(top_right - top_left))
    width_bottom = float(np.linalg.norm(bottom_right - bottom_left))
    height_left = float(np.linalg.norm(bottom_left - top_left))
    height_right = float(np.linalg.norm(bottom_right - top_right))

    output_width = round(max(width_top, width_bottom))
    output_height = round(max(height_left, height_right))

    if output_width < 3 or output_height < 3:
        logger.warning(
            "Detected document corners produce a degenerate %dx%d output; "
            "skipping perspective correction.",
            output_width,
            output_height,
        )
        return image

    src_points = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
    dst_points = np.array(
        [
            [0, 0],
            [output_width - 1, 0],
            [output_width - 1, output_height - 1],
            [0, output_height - 1],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(
        image,
        matrix,
        (output_width, output_height),
        flags=config.perspective_interpolation,
    )

    logger.info(
        "Corrected document perspective: cropped/warped to %dx%d.",
        output_width,
        output_height,
    )
    return warped


def denoise_image(
    image: np.ndarray,
    config: PreprocessingConfig = PREPROCESSING_CONFIG,
) -> np.ndarray:
    """
    Reduce noise using OpenCV's Non-Local Means denoising.

    Non-Local Means averages each pixel with similar-looking patches
    elsewhere in the image rather than blurring uniformly across a fixed
    neighborhood, which is what lets it suppress speckle/scan noise
    while keeping edges (and thin character strokes) comparatively
    intact — a Gaussian or median blur strong enough to remove the same
    noise would blur strokes noticeably more. Filter strength is kept
    moderate (see :class:`PreprocessingConfig`) specifically to avoid
    over-smoothing text.

    Args:
        image: Grayscale or RGB(A) uint8 image array.
        config: Denoising strength/window settings.

    Returns:
        A new, denoised image array (dimensions and channel count unchanged).
    """
    _validate_image(image)

    if image.ndim == 2:
        return cv2.fastNlMeansDenoising(
            image,
            None,
            h=config.denoise_h_luminance,
            templateWindowSize=config.denoise_template_window_size,
            searchWindowSize=config.denoise_search_window_size,
        )

    has_alpha = image.shape[2] == 4
    rgb = image[:, :, :3]
    denoised_rgb = cv2.fastNlMeansDenoisingColored(
        rgb,
        None,
        h=config.denoise_h_luminance,
        hColor=config.denoise_h_color,
        templateWindowSize=config.denoise_template_window_size,
        searchWindowSize=config.denoise_search_window_size,
    )
    if not has_alpha:
        return denoised_rgb
    return np.concatenate([denoised_rgb, image[:, :, 3:4]], axis=2)


def deskew_image(
    image: np.ndarray,
    angle_degrees: float,
    config: PreprocessingConfig = PREPROCESSING_CONFIG,
) -> np.ndarray:
    """
    Rotate a page to correct skew, using the angle Phase 2 already measured.

    Uses the exact same angle convention as
    :func:`src.quality.image_quality.assess_skew`: rotating the image by
    ``angle_degrees`` via ``cv2.getRotationMatrix2D`` directly (with no
    sign flip) undoes that measured skew. This has been verified against
    ``assess_skew``'s own geometry (rotating a page by +N degrees before
    measurement yields a measured angle of -N degrees, so applying the
    measured angle directly, unnegated, brings the page back to level)
    and works the same way for positive skew, negative skew, and large
    skew angles.

    An angle smaller than ``config.deskew_min_angle_degrees`` is treated
    as already-straight and the image is returned unchanged, so a page
    that is merely at the edge of Phase 2's threshold isn't needlessly
    blurred by rotation interpolation.

    Args:
        image: Grayscale or RGB(A) uint8 image array.
        angle_degrees: Skew angle in degrees, as measured by
            :func:`~src.quality.image_quality.assess_skew` (e.g. from
            ``PageQualityReport.skew.angle_degrees``).
        config: Interpolation and border-fill settings.

    Returns:
        The rotated image (same dimensions, corners filled with
        ``config.deskew_border_value``), or the original array unchanged
        if the angle is negligible.
    """
    _validate_image(image)

    if abs(angle_degrees) < config.deskew_min_angle_degrees:
        return image

    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)

    border_value = (
        config.deskew_border_value
        if image.ndim == 2
        else (config.deskew_border_value,) * image.shape[2]
    )

    logger.info("Deskewing page by %.2f degrees", angle_degrees)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=config.deskew_interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def enhance_contrast(
    image: np.ndarray,
    config: PreprocessingConfig = PREPROCESSING_CONFIG,
) -> np.ndarray:
    """
    Improve local contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).

    CLAHE equalizes contrast within small tiles rather than across the
    whole page, so it can brighten faint text in one region without
    blowing out an already-clear region elsewhere — a document-friendly
    property that a single global equalization doesn't have. The "limited"
    part caps how much any tile can be stretched, which keeps noise from
    being amplified in flat (background) areas.

    For a color image, CLAHE is applied to the L (lightness) channel of
    LAB color space only, then merged back, so color information (the A/B
    channels) is left untouched and no hue shift is introduced.

    Args:
        image: Grayscale or RGB(A) uint8 image array.
        config: CLAHE clip limit and tile grid size.

    Returns:
        A new, contrast-enhanced image array (dimensions and channel
        count unchanged).
    """
    _validate_image(image)

    clahe = cv2.createCLAHE(
        clipLimit=config.clahe_clip_limit,
        tileGridSize=(config.clahe_tile_grid_size, config.clahe_tile_grid_size),
    )

    if image.ndim == 2:
        return clahe.apply(image)

    has_alpha = image.shape[2] == 4
    rgb = image[:, :, :3]

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    l_equalized = clahe.apply(l_channel)
    equalized_lab = cv2.merge((l_equalized, a_channel, b_channel))
    equalized_rgb = cv2.cvtColor(equalized_lab, cv2.COLOR_LAB2RGB)

    if not has_alpha:
        return equalized_rgb
    return np.concatenate([equalized_rgb, image[:, :, 3:4]], axis=2)


def threshold_image(
    image: np.ndarray,
    config: PreprocessingConfig = PREPROCESSING_CONFIG,
) -> np.ndarray:
    """
    Binarize a page into foreground (text) and background.

    Two classical methods are supported via ``config.threshold_method``:

    * ``"otsu"`` (default): a single global threshold chosen automatically
      from the image's histogram. Simple, deterministic, and consistent
      with the OTSU method Phase 2's ``assess_skew`` already uses
      internally — appropriate when illumination is reasonably even
      across the page.
    * ``"adaptive"``: a threshold computed per local neighborhood, better
      suited to pages with uneven lighting or shadows that a single
      global threshold can't handle well.

    Unlike the other operations, this always returns a single-channel
    image: binarization is inherently a channel-reducing operation, so
    "preserve dimensions" here means preserving height/width, not channel
    count.

    Args:
        image: Grayscale or RGB(A) uint8 image array.
        config: Selects the thresholding method and its parameters.

    Returns:
        A single-channel uint8 binary image (values 0 or 255).

    Raises:
        PreprocessingError: If ``config.threshold_method`` is not a
            recognized method.
    """
    _validate_image(image)
    gray = _to_grayscale(image)

    if config.threshold_method == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    if config.threshold_method == "adaptive":
        block_size = config.adaptive_block_size
        if block_size % 2 == 0:
            block_size += 1  # cv2.adaptiveThreshold requires an odd block size
        return cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            config.adaptive_c,
        )

    raise PreprocessingError(
        f"Unknown threshold_method '{config.threshold_method}'; expected 'otsu' or 'adaptive'."
    )


# --------------------------------------------------------------------------
# High-level pipeline
# --------------------------------------------------------------------------


def preprocess_page(
    image: np.ndarray,
    quality_report: PageQualityReport,
    config: PreprocessingConfig = PREPROCESSING_CONFIG,
) -> ProcessedPage:
    """
    Perform the preprocessing operations Phase 2 recommended for a page.

    Operations are applied in a fixed order regardless of the order they
    appear in ``quality_report.recommendations``:

        Resize -> Perspective correction -> Denoise -> Deskew ->
        Contrast enhancement -> Thresholding

    This order is chosen so that each step gets the best possible input:
    resizing first means every later step (and any later re-inspection)
    works at the intended resolution; perspective correction runs next,
    before any pixel-cleanup step, so denoise/deskew/contrast/threshold
    all operate on just the cropped, straightened page rather than
    partly on background that's about to be discarded anyway; denoising
    before deskewing gives the rotation's interpolation clean input
    instead of amplifying noise into new artifacts along edges;
    deskewing before contrast enhancement means CLAHE's tiles are
    aligned with the page's true content instead of a rotated version
    of it; and thresholding runs last because it is a one-way,
    irreversible step (turning grayscale into binary) that should only
    happen once every earlier, reversible cleanup step has already run.

    Perspective correction is a special case worth calling out: Phase 2
    detects the document corners once, up front, on the *original*
    image — but if RESIZE already ran, ``current`` is no longer that
    original image's pixel grid. This function rescales
    ``quality_report.perspective.corners`` by the same factor RESIZE
    applied before handing them to :func:`correct_perspective`, so the
    warp still targets the right pixels regardless of whether resizing
    happened first.

    An operation is only invoked if it appears in
    ``quality_report.recommendations`` — if Phase 2 recommends nothing,
    ``image`` is returned unmodified. ``operations_applied`` reflects
    what actually changed the image, not merely what was recommended:
    :func:`resize_image` and :func:`deskew_image` can each decide, from
    the image/angle itself, that no change is needed (e.g. an image
    already at the target size), and such no-ops are not recorded as
    applied.

    Args:
        image: Rendered page image, as an RGB(A) or grayscale uint8
            numpy array (the same format Phase 2's ``assess_page_quality``
            accepts).
        quality_report: The Phase 2 :class:`PageQualityReport` for this
            same image, whose ``recommendations`` drive which operations
            run and whose ``resolution.dpi`` / ``skew.angle_degrees``
            parameterize them.
        config: Preprocessing settings for the individual operations.

    Returns:
        A :class:`ProcessedPage` containing the processed image and the
        list of operations actually applied.
    """
    _validate_image(image)

    recommended = {rec.operation for rec in quality_report.recommendations}
    current = image
    applied: List[PreprocessingOperation] = []

    if PreprocessingOperation.RESIZE in recommended:
        resized = resize_image(current, current_dpi=quality_report.resolution.dpi, config=config)
        if resized is not current:
            current = resized
            applied.append(PreprocessingOperation.RESIZE)

    if PreprocessingOperation.PERSPECTIVE_CORRECTION in recommended:
        corners = quality_report.perspective.corners
        if corners is not None and current.shape[:2] != image.shape[:2]:
            # RESIZE already changed current's pixel grid; rescale the
            # corners (measured by Phase 2 on the original image) to match.
            scale_x = current.shape[1] / image.shape[1]
            scale_y = current.shape[0] / image.shape[0]
            corners = [(x * scale_x, y * scale_y) for (x, y) in corners]
        corrected = correct_perspective(current, corners=corners, config=config)
        if corrected is not current:
            current = corrected
            applied.append(PreprocessingOperation.PERSPECTIVE_CORRECTION)

    if PreprocessingOperation.DENOISE in recommended:
        current = denoise_image(current, config=config)
        applied.append(PreprocessingOperation.DENOISE)

    if PreprocessingOperation.DESKEW in recommended:
        deskewed = deskew_image(current, angle_degrees=quality_report.skew.angle_degrees, config=config)
        if deskewed is not current:
            current = deskewed
            applied.append(PreprocessingOperation.DESKEW)

    if PreprocessingOperation.CONTRAST_ENHANCEMENT in recommended:
        current = enhance_contrast(current, config=config)
        applied.append(PreprocessingOperation.CONTRAST_ENHANCEMENT)

    if PreprocessingOperation.THRESHOLDING in recommended:
        current = threshold_image(current, config=config)
        applied.append(PreprocessingOperation.THRESHOLDING)

    logger.info(
        "Page %d preprocessing complete: applied=%s",
        quality_report.page_number,
        [op.value for op in applied] or ["NONE"],
    )

    return ProcessedPage(
        page_number=quality_report.page_number,
        image=current,
        operations_applied=applied,
    )
