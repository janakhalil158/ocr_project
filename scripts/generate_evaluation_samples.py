"""
Generate the initial synthetic OCR evaluation samples.

These samples are rendered from text this script *defines*, so the
ground truth is known by construction rather than transcribed from OCR
output — which is the whole point: ground truth produced by running
OCR would make the benchmark circular and meaningless.

The clean English/Arabic/mixed samples establish a best-case ceiling;
the blurry/low-contrast/skewed variants apply a controlled degradation
to the *same* source text, so the CER/WER difference isolates the
effect of that specific degradation.

Run from the project root::

    python scripts/generate_evaluation_samples.py
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import arabic_reshaper
    from PIL import features as _pil_features

    ARABIC_AVAILABLE = True
    #: Pillow built with RAQM does its own bidi reordering during layout.
    RAQM_AVAILABLE = _pil_features.check("raqm")
except ImportError:
    ARABIC_AVAILABLE = False
    RAQM_AVAILABLE = False


def _shape_arabic(line: str) -> str:
    """
    Prepare an Arabic string for rendering.

    Applies letter-joining (reshaping) always. Applies bidi reordering
    ONLY when Pillow lacks RAQM — otherwise Pillow reorders the line
    itself and a second reordering here would render it reversed.
    """
    reshaped = arabic_reshaper.reshape(line)
    if RAQM_AVAILABLE:
        return reshaped
    from bidi.algorithm import get_display

    return get_display(reshaped)

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "data" / "evaluation"
IMAGES_DIR = EVAL_DIR / "images"
TRUTH_DIR = EVAL_DIR / "ground_truth"

LATIN_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
ARABIC_FONT = "/usr/share/fonts/truetype/kacst/KacstBook.ttf"

ENGLISH_TEXT = [
    "Certificate of Completion",
    "This is to certify that the named student",
    "has successfully completed the training course",
    "in document processing and analysis.",
    "Issued on 14 March 2026 by the Institute.",
]

ARABIC_TEXT = [
    "شهادة إتمام الدورة",
    "تشهد المؤسسة بأن الطالب",
    "قد أتم الدورة التدريبية بنجاح",
]

MIXED_TEXT = [
    "Certificate of Completion",
    "شهادة إتمام الدورة",
    "Issued by the Institute",
]


def _render_lines(lines, font_path, font_size=40, width=1100, line_height=70, rtl=False):
    """
    Render text lines onto a white page image.

    Arabic note: only ``arabic_reshaper.reshape()`` is applied, and NOT
    ``bidi.algorithm.get_display()``. Pillow here is built with RAQM,
    which performs bidi reordering itself during layout; applying
    get_display() first would reorder the string a second time and
    render the line character-reversed. That was verified empirically —
    with the double reordering, Tesseract returned the exact character
    reverse of the source text (CER ~0.83); with reshape alone it
    returns the source text exactly (CER 0.0).

    If you run this generator on a Pillow build WITHOUT RAQM
    (``PIL.features.check("raqm")`` is False), Pillow will not reorder
    or shape, and you would then need get_display() plus reshaping to
    get a correct image. Check that flag before trusting Arabic output.
    """
    height = line_height * len(lines) + 80
    image = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path, font_size)

    for i, line in enumerate(lines):
        rendered = line
        if rtl and ARABIC_AVAILABLE:
            rendered = _shape_arabic(line)
        y = 40 + i * line_height
        if rtl:
            # Right-align RTL text.
            text_width = draw.textlength(rendered, font=font)
            draw.text((width - 40 - text_width, y), rendered, font=font, fill=(0, 0, 0))
        else:
            draw.text((40, y), rendered, font=font, fill=(0, 0, 0))

    return np.array(image)


def _render_mixed(lines, font_size=40, width=1100, line_height=70):
    """Render mixed Arabic/English, choosing a font per line by script."""
    height = line_height * len(lines) + 80
    image = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    latin_font = ImageFont.truetype(LATIN_FONT, font_size)
    arabic_font = ImageFont.truetype(ARABIC_FONT, font_size)

    for i, line in enumerate(lines):
        is_arabic = any("\u0600" <= ch <= "\u06FF" for ch in line)
        y = 40 + i * line_height
        if is_arabic and ARABIC_AVAILABLE:
            rendered = _shape_arabic(line)
            text_width = draw.textlength(rendered, font=arabic_font)
            draw.text((width - 40 - text_width, y), rendered, font=arabic_font, fill=(0, 0, 0))
        else:
            draw.text((40, y), line, font=latin_font, fill=(0, 0, 0))

    return np.array(image)


def _blur(image, ksize=5):
    return cv2.GaussianBlur(image, (ksize, ksize), 0)


def _low_contrast(image, factor=0.35):
    """Compress the dynamic range toward mid-grey."""
    mid = 128.0
    return np.clip((image.astype(np.float32) - mid) * factor + mid, 0, 255).astype(np.uint8)


def _skew(image, angle=5.0):
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(
        image, matrix, (w, h), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255),
    )


def _save(sample_id, image, text_lines):
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    TRUTH_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(IMAGES_DIR / f"{sample_id}.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    (TRUTH_DIR / f"{sample_id}.txt").write_text("\n".join(text_lines), encoding="utf-8")


def main() -> None:
    samples = []

    english_clean = _render_lines(ENGLISH_TEXT, LATIN_FONT)
    _save("english_clean_001", english_clean, ENGLISH_TEXT)
    samples.append(("english_clean_001", "eng", "clean"))

    _save("english_blurry_001", _blur(english_clean), ENGLISH_TEXT)
    samples.append(("english_blurry_001", "eng", "blurry"))

    _save("english_low_contrast_001", _low_contrast(english_clean), ENGLISH_TEXT)
    samples.append(("english_low_contrast_001", "eng", "low_contrast"))

    _save("english_skewed_001", _skew(english_clean), ENGLISH_TEXT)
    samples.append(("english_skewed_001", "eng", "skewed"))

    if ARABIC_AVAILABLE and Path(ARABIC_FONT).exists():
        arabic_clean = _render_lines(ARABIC_TEXT, ARABIC_FONT, rtl=True)
        _save("arabic_clean_001", arabic_clean, ARABIC_TEXT)
        samples.append(("arabic_clean_001", "ara", "arabic"))

        _save("arabic_blurry_001", _blur(arabic_clean), ARABIC_TEXT)
        samples.append(("arabic_blurry_001", "ara", "blurry"))

        mixed = _render_mixed(MIXED_TEXT)
        _save("mixed_language_001", mixed, MIXED_TEXT)
        samples.append(("mixed_language_001", "ara+eng", "mixed_language"))

    manifest = {
        "description": (
            "OCR evaluation dataset. Ground truth for these synthetic samples is "
            "known by construction (the source text was rendered to the image by "
            "scripts/generate_evaluation_samples.py), NOT transcribed from OCR "
            "output. Real-document samples added later must have their ground "
            "truth transcribed and verified by a human."
        ),
        "samples": [
            {
                "id": sample_id,
                "image": f"images/{sample_id}.png",
                "ground_truth": f"ground_truth/{sample_id}.txt",
                "language": language,
                "category": category,
                "notes": "synthetic; ground truth known by construction",
            }
            for sample_id, language, category in samples
        ],
    }

    (EVAL_DIR / "dataset.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Generated {len(samples)} evaluation sample(s).")
    for sample_id, language, category in samples:
        print(f"  {sample_id}  [{language}/{category}]")


if __name__ == "__main__":
    main()
