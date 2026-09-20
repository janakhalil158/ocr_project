"""
Routing logic.

Given a document's classification, decides what the NEXT pipeline
action should be. This module only makes the decision — it does not
execute OCR, text extraction, or any other downstream step. Those are
later phases.
"""

from __future__ import annotations

from src.pdf.pdf_type_detector import DocumentType
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RoutingError(Exception):
    """Raised when a document type cannot be routed."""


# Explicit, single source of truth for the routing table required by Phase 1.
_ROUTING_TABLE: dict[DocumentType, str] = {
    DocumentType.TEXT_PDF: "extract_text",
    DocumentType.SCANNED_PDF: "render_pages_for_ocr",
    DocumentType.MIXED_PDF: "process_pages_individually",
}


def route(document_type: DocumentType) -> str:
    """
    Determine the next pipeline action for a classified document.

    Args:
        document_type: The overall classification of the PDF, as
            produced by :func:`src.pdf.pdf_type_detector.detect_pdf_type`.

    Returns:
        A string identifying the next intended action:
        ``"extract_text"``, ``"render_pages_for_ocr"``, or
        ``"process_pages_individually"``.

    Raises:
        RoutingError: If ``document_type`` has no known routing rule.
    """
    try:
        action = _ROUTING_TABLE[document_type]
    except KeyError as exc:
        logger.error("No routing rule defined for document type: %s", document_type)
        raise RoutingError(f"No routing rule defined for document type: {document_type}") from exc

    logger.info("Routing decision for %s -> %s", document_type.value, action)
    return action
