"""
Phase 5: OCR accuracy metrics (CER / WER).

Pure, dependency-free text-comparison functions used by the OCR
evaluator. Nothing here knows about OCR engines, images, or the
pipeline — these are just string metrics, which keeps them trivially
testable and equally applicable to Tesseract today and PaddleOCR later.

Why these metrics at all: an engine's own *confidence* score says how
sure the engine was, not whether it was right. A confidently-misread
word scores high confidence and is still wrong. CER/WER compare the
output against verified ground truth, which is the only way to make an
objective Tesseract-vs-PaddleOCR claim.
"""

from __future__ import annotations

import unicodedata
from typing import List, Sequence


def levenshtein_distance(reference: Sequence, hypothesis: Sequence) -> int:
    """
    Edit distance between two sequences (characters or word lists).

    Counts substitutions, insertions, and deletions with equal weight
    of 1. Implemented with a rolling two-row table rather than a full
    matrix, so memory is O(min(len)) — enough for page-length text
    without pulling in an external dependency.
    """
    if reference == hypothesis:
        return 0
    if len(reference) == 0:
        return len(hypothesis)
    if len(hypothesis) == 0:
        return len(reference)

    # Iterate over the shorter sequence in the inner dimension.
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference

    previous_row = list(range(len(hypothesis) + 1))
    for i, ref_item in enumerate(reference, start=1):
        current_row = [i]
        for j, hyp_item in enumerate(hypothesis, start=1):
            cost = 0 if ref_item == hyp_item else 1
            current_row.append(
                min(
                    previous_row[j] + 1,        # deletion
                    current_row[j - 1] + 1,     # insertion
                    previous_row[j - 1] + cost,  # substitution
                )
            )
        previous_row = current_row

    return previous_row[-1]


def normalize_text(text: str) -> str:
    """
    Normalize text before CER/WER comparison.

    Exactly what is done — and deliberately nothing more:

    1. Unicode NFC normalization, so visually identical text that
       happens to use decomposed code points (common with Arabic
       diacritics and accented Latin) compares equal instead of
       registering as errors.
    2. Line breaks (``\\n``, ``\\r``) collapsed to single spaces, so a
       difference in *where* the engine broke lines isn't scored as a
       text error — line-structure quality is a separate concern from
       character accuracy.
    3. Runs of whitespace collapsed to a single space.
    4. Leading/trailing whitespace stripped.

    What is deliberately NOT done, because it would flatter the engine
    and corrupt the measurement:

    * No lowercasing — case errors are real OCR errors.
    * No punctuation stripping.
    * No spell correction or dictionary snapping of any kind.
    * No Arabic-specific rewriting: no diacritic (tashkeel) removal,
      no alef/hamza unification, no tatweel stripping, no
      Arabic-Indic-to-ASCII digit mapping. Those transformations
      destroy real character distinctions and would silently inflate
      Arabic scores.

    Returns:
        The normalized string. Arabic, and every other script, passes
        through with its characters intact.
    """
    if not text:
        return ""

    normalized = unicodedata.normalize("NFC", text)
    normalized = normalized.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    normalized = " ".join(normalized.split())
    return normalized


def tokenize_words(text: str) -> List[str]:
    """Split normalized text into whitespace-delimited word tokens."""
    normalized = normalize_text(text)
    return normalized.split() if normalized else []


def character_error_rate(reference: str, hypothesis: str) -> float:
    """
    Character Error Rate: ``edit_distance(ref, hyp) / len(ref)``.

    Both inputs are passed through :func:`normalize_text` first.

    Empty-reference behavior (documented rather than left to divide by
    zero): an empty reference with an empty hypothesis scores ``0.0``
    (nothing expected, nothing produced — a perfect match). An empty
    reference with a non-empty hypothesis scores ``1.0``, treating
    everything the engine invented as error. ``1.0`` is used rather
    than the unbounded ``len(hyp)/0`` so the value stays on the same
    scale as every other sample and can be averaged safely.

    Note the rate is NOT capped at 1.0 in the normal case: an engine
    that hallucinates far more text than the reference contains can
    legitimately score above 1.0, and hiding that would understate a
    real failure.
    """
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)

    if not ref:
        return 0.0 if not hyp else 1.0

    return levenshtein_distance(ref, hyp) / len(ref)


def word_error_rate(reference: str, hypothesis: str) -> float:
    """
    Word Error Rate: ``word_edit_distance(ref, hyp) / len(ref_words)``.

    Word-level analogue of :func:`character_error_rate`, using the same
    normalization and the same documented empty-reference convention
    (empty/empty → ``0.0``; empty reference with output → ``1.0``).
    """
    ref_words = tokenize_words(reference)
    hyp_words = tokenize_words(hypothesis)

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    return levenshtein_distance(ref_words, hyp_words) / len(ref_words)
