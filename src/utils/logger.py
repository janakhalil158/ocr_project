"""
Centralized logging configuration for the AI Document Processing pipeline.

Provides a single `get_logger` factory so every module logs in a
consistent format, instead of each module configuring logging itself.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"

_configured = False


def _configure_root(level: int) -> None:
    """Configure the root logger exactly once with a stream handler."""
    global _configured
    if _configured:
        return

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT))

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)

    _configured = True


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Return a configured logger for the given module name.

    Args:
        name: Typically ``__name__`` of the calling module.
        level: Optional log level override for this specific logger
            (e.g. ``logging.DEBUG``). Defaults to the root logger's level.

    Returns:
        A ready-to-use ``logging.Logger`` instance.
    """
    _configure_root(level or logging.INFO)

    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)

    return logger
