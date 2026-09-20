"""
Lightweight row / column / cell reconstruction for PDF text.

This module does not claim to be a full computer-vision table detector.
It reconstructs structured rows and cells from positional text whenever
the PDF contains usable word coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from src.pdf.text_extractor import PDFWord


DEFAULT_Y_TOLERANCE = 6.0
DEFAULT_COLUMN_TOLERANCE = 15.0


@dataclass(frozen=True)
class LayoutCell:
    """A reconstructed table cell."""

    column: int
    text: str
    x0: float
    x1: float


@dataclass(frozen=True)
class LayoutRow:
    """One visual row."""

    y: float
    words: Tuple[PDFWord, ...]

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)


@dataclass(frozen=True)
class PageLayout:
    """Reconstructed page layout."""

    rows: Tuple[LayoutRow, ...]
    estimated_columns: Tuple[float, ...]
    cells: Tuple[Tuple[LayoutCell, ...], ...]

    @property
    def word_count(self) -> int:
        return sum(len(row.words) for row in self.rows)


def _group_into_rows(
    words: Tuple[PDFWord, ...],
    y_tolerance: float,
) -> Tuple[LayoutRow, ...]:

    if not words:
        return ()

    ordered = sorted(
        words,
        key=lambda w: (w.y0, w.x0),
    )

    row_words: List[List[PDFWord]] = []
    row_reference_y: List[float] = []

    for word in ordered:
        best_row = None
        best_distance = None

        for idx, ref_y in enumerate(row_reference_y):
            distance = abs(word.y0 - ref_y)

            if distance <= y_tolerance:
                if best_distance is None or distance < best_distance:
                    best_row = idx
                    best_distance = distance

        if best_row is None:
            row_words.append([word])
            row_reference_y.append(word.y0)
        else:
            row_words[best_row].append(word)

            # Slowly move the reference toward the actual row.
            row_reference_y[best_row] = sum(
                w.y0 for w in row_words[best_row]
            ) / len(row_words[best_row])

    rows = []

    for words_in_row in row_words:
        sorted_words = tuple(
            sorted(words_in_row, key=lambda w: w.x0)
        )

        avg_y = sum(
            w.y0 for w in sorted_words
        ) / len(sorted_words)

        rows.append(
            LayoutRow(
                y=avg_y,
                words=sorted_words,
            )
        )

    return tuple(
        sorted(rows, key=lambda row: row.y)
    )


def _estimate_columns(
    rows: Tuple[LayoutRow, ...],
    column_tolerance: float,
) -> Tuple[float, ...]:

    positions = sorted(
        w.x0
        for row in rows
        for w in row.words
    )

    if not positions:
        return ()

    clusters: List[List[float]] = []

    for x in positions:
        if (
            clusters
            and abs(x - clusters[-1][-1]) <= column_tolerance
        ):
            clusters[-1].append(x)
        else:
            clusters.append([x])

    return tuple(
        sum(cluster) / len(cluster)
        for cluster in clusters
    )


def _assign_column(
    x0: float,
    columns: Tuple[float, ...],
) -> int:

    if not columns:
        return 0

    return min(
        range(len(columns)),
        key=lambda i: abs(columns[i] - x0),
    )


def _reconstruct_cells(
    rows: Tuple[LayoutRow, ...],
    columns: Tuple[float, ...],
) -> Tuple[Tuple[LayoutCell, ...], ...]:

    reconstructed = []

    for row in rows:

        cell_words: dict[int, List[PDFWord]] = {}

        for word in row.words:
            column = _assign_column(
                word.x0,
                columns,
            )

            cell_words.setdefault(
                column,
                [],
            ).append(word)

        cells = []

        for column in sorted(cell_words):

            words = sorted(
                cell_words[column],
                key=lambda w: w.x0,
            )

            cells.append(
                LayoutCell(
                    column=column,
                    text=" ".join(
                        word.text
                        for word in words
                    ),
                    x0=min(
                        word.x0
                        for word in words
                    ),
                    x1=max(
                        word.x1
                        for word in words
                    ),
                )
            )

        reconstructed.append(tuple(cells))

    return tuple(reconstructed)


def analyze_layout(
    words: Tuple[PDFWord, ...],
    y_tolerance: float = DEFAULT_Y_TOLERANCE,
    column_tolerance: float = DEFAULT_COLUMN_TOLERANCE,
) -> PageLayout:

    rows = _group_into_rows(
        tuple(words),
        y_tolerance,
    )

    columns = _estimate_columns(
        rows,
        column_tolerance,
    )

    cells = _reconstruct_cells(
        rows,
        columns,
    )

    return PageLayout(
        rows=rows,
        estimated_columns=columns,
        cells=cells,
    )

