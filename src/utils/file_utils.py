"""
Small, dependency-free filesystem helper functions shared across modules.
"""

from __future__ import annotations

from pathlib import Path


class FileValidationError(Exception):
    """Raised when a file fails basic existence/type validation."""


def validate_file_exists(file_path: str | Path) -> Path:
    """
    Validate that a path exists and points to a file (not a directory).

    Args:
        file_path: Path to the file to validate.

    Returns:
        The resolved ``Path`` object.

    Raises:
        FileValidationError: If the path does not exist or is not a file.
    """
    path = Path(file_path).expanduser().resolve()

    if not path.exists():
        raise FileValidationError(f"File does not exist: {path}")
    if not path.is_file():
        raise FileValidationError(f"Path is not a file: {path}")

    return path


def validate_extension(file_path: str | Path, expected_extension: str) -> Path:
    """
    Validate that a file has the expected extension (case-insensitive).

    Args:
        file_path: Path to the file to validate.
        expected_extension: Expected extension, with or without a leading dot
            (e.g. ``".pdf"`` or ``"pdf"``).

    Returns:
        The resolved ``Path`` object.

    Raises:
        FileValidationError: If the extension does not match.
    """
    path = Path(file_path)
    expected = expected_extension if expected_extension.startswith(".") else f".{expected_extension}"

    if path.suffix.lower() != expected.lower():
        raise FileValidationError(
            f"Expected a '{expected}' file but got '{path.suffix}': {path}"
        )

    return path


def get_file_size_bytes(file_path: str | Path) -> int:
    """Return the size of a file in bytes."""
    return Path(file_path).stat().st_size


def ensure_directory(directory: str | Path) -> Path:
    """Create a directory (and parents) if it does not already exist."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path
