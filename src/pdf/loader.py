"""
PDF loading and metadata extraction.

Responsible for safely opening PDF files and exposing basic metadata
(filename, size, page count) without making any assumptions about the
document's content type. All downstream modules (type detection,
rendering) should receive a PDF via this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pymupdf as fitz  # PyMuPDF (using the non-deprecated import name)

from src.utils.file_utils import (
    FileValidationError,
    get_file_size_bytes,
    validate_extension,
    validate_file_exists,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PDFLoadError(Exception):
    """Raised when a PDF file cannot be validated or opened."""


@dataclass(frozen=True)
class PDFMetadata:
    """Basic metadata describing a loaded PDF."""

    filename: str
    file_path: Path
    file_size_bytes: int
    num_pages: int


def validate_pdf(file_path: str | Path) -> Path:
    """
    Validate that a path exists and is a ``.pdf`` file.

    This only checks the extension and existence; it does not verify
    that the file is structurally a valid PDF (see :func:`load_pdf`
    for that).

    Args:
        file_path: Path to the candidate PDF file.

    Returns:
        The resolved, validated ``Path``.

    Raises:
        PDFLoadError: If the file does not exist or isn't a ``.pdf`` file.
    """
    try:
        path = validate_file_exists(file_path)
        path = validate_extension(path, ".pdf")
    except FileValidationError as exc:
        logger.error("PDF validation failed: %s", exc)
        raise PDFLoadError(str(exc)) from exc

    return path


def load_pdf(file_path: str | Path) -> fitz.Document:
    """
    Safely open a PDF file with PyMuPDF.

    Args:
        file_path: Path to the PDF file.

    Returns:
        An opened ``fitz.Document`` instance. Callers are responsible
        for closing it (or use it as a context manager).

    Raises:
        PDFLoadError: If the file fails validation, is corrupted, is
            encrypted without a usable password, or otherwise cannot
            be opened by PyMuPDF.
    """
    path = validate_pdf(file_path)

    try:
        document = fitz.open(path)
    except Exception as exc:  # PyMuPDF raises varied exception types for bad PDFs
        logger.error("Failed to open PDF '%s': %s", path, exc)
        raise PDFLoadError(f"Could not open PDF '{path.name}': {exc}") from exc

    if document.is_encrypted and not document.authenticate(""):
        document.close()
        logger.error("PDF '%s' is encrypted and could not be opened", path)
        raise PDFLoadError(f"PDF '{path.name}' is encrypted and requires a password.")

    if document.page_count == 0:
        document.close()
        logger.error("PDF '%s' contains zero pages", path)
        raise PDFLoadError(f"PDF '{path.name}' is invalid or contains no pages.")

    logger.info("Successfully opened PDF '%s' (%d pages)", path.name, document.page_count)
    return document


def get_pdf_metadata(file_path: str | Path) -> PDFMetadata:
    """
    Load a PDF and return its basic metadata.

    Opens and closes the document internally; use :func:`load_pdf`
    directly if you need to keep the document open for further work.

    Args:
        file_path: Path to the PDF file.

    Returns:
        A :class:`PDFMetadata` instance.

    Raises:
        PDFLoadError: If the PDF cannot be validated or opened.
    """
    path = validate_pdf(file_path)
    document = load_pdf(path)

    try:
        metadata = PDFMetadata(
            filename=path.name,
            file_path=path,
            file_size_bytes=get_file_size_bytes(path),
            num_pages=document.page_count,
        )
    finally:
        document.close()

    logger.debug("Extracted metadata for '%s': %s", path.name, metadata)
    return metadata
