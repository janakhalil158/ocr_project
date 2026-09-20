from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import fitz

from src.pdf.loader import PDFLoadError, load_pdf
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class PDFWord:
    """A word extracted from a PDF together with its PDF coordinates."""

    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    block_number: int = 0
    line_number: int = 0
    word_number: int = 0


@dataclass(frozen=True)
class PDFPageText:
    """Native text extraction result for one PDF page."""

    page_number: int
    text: str
    words: Tuple[PDFWord, ...]
    width: float
    height: float

    @property
    def character_count(self) -> int:
        return len(self.text)


@dataclass(frozen=True)
class PDFTextExtractionResult:
    """Native text extraction result for an entire PDF."""

    pages: Tuple[PDFPageText, ...]

    @property
    def total_characters(self) -> int:
        return sum(page.character_count for page in self.pages)


# Backward-compatible alias used by the existing pipeline.
PDFTextExtraction = PDFTextExtractionResult


def extract_text_from_page(page) -> PDFPageText:
    """
    Extract native text and positional words from one PDF page.
    """

    text = page.get_text("text").strip()

    raw_words = page.get_text("words")

    words = tuple(
        PDFWord(
            text=str(word[4]),
            x0=float(word[0]),
            y0=float(word[1]),
            x1=float(word[2]),
            y1=float(word[3]),
            block_number=int(word[5]),
            line_number=int(word[6]),
            word_number=int(word[7]),
        )
        for word in raw_words
        if str(word[4]).strip()
    )

    return PDFPageText(
        page_number=page.number + 1,
        text=text,
        words=words,
        width=float(page.rect.width),
        height=float(page.rect.height),
    )


def extract_text_from_pdf(
    pdf_path: str | Path,
) -> PDFTextExtractionResult:
    """
    Extract native text from every page of a PDF.
    """

    pdf_path = Path(pdf_path)

    document = load_pdf(pdf_path)

    try:
        pages = tuple(
            extract_text_from_page(page)
            for page in document
        )

        return PDFTextExtractionResult(pages=pages)

    finally:
        document.close()


def extract_text_from_pdf_safe(
    pdf_path: str | Path,
) -> PDFTextExtractionResult | None:
    """
    Safely extract native PDF text.

    Returns None if extraction fails.
    """

    try:
        result = extract_text_from_pdf(pdf_path)

        logger.info(
            "Extracted native text from %d page(s), %d total characters",
            len(result.pages),
            result.total_characters,
        )

        return result

    except Exception as exc:
        logger.exception(
            "Native PDF text extraction failed: %s",
            exc,
        )
        return None


def is_text_quality_good(text: str) -> bool:
    """
    Check whether native PDF text appears reliable.

    This is especially useful for PDFs whose Arabic font encoding
    produces broken Unicode characters during native extraction.
    """

    if not text or len(text.strip()) < 10:
        return False

    suspicious_chars = {
        "�",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
    }

    if any(char in text for char in suspicious_chars):
        return False

    private_use_count = sum(
        1
        for char in text
        if "\ue000" <= char <= "\uf8ff"
    )

    if private_use_count > 0:
        return False

    arabic_words = []
    isolated_arabic = 0

    for word in text.split():
        arabic_chars = [
            char
            for char in word
            if "\u0600" <= char <= "\u06FF"
        ]

        if arabic_chars:
            arabic_words.append(word)

            if len(arabic_chars) == 1:
                isolated_arabic += 1

    if len(arabic_words) >= 10:
        isolated_ratio = isolated_arabic / len(arabic_words)

        if isolated_ratio > 0.30:
            return False

    return True