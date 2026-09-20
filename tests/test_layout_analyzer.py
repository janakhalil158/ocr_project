"""
Tests for src/pdf/layout_analyzer.py.

Builds PDFWord instances directly (no PDF/OCR involved) since this
module operates purely on already-extracted positional word data.
"""

from __future__ import annotations

from src.pdf.layout_analyzer import PageLayout, analyze_layout
from src.pdf.text_extractor import PDFWord


def _word(text: str, x0: float, y0: float, block: int = 0, line: int = 0, word_no: int = 0) -> PDFWord:
    return PDFWord(
        text=text,
        x0=x0,
        y0=y0,
        x1=x0 + 20.0,
        y1=y0 + 10.0,
        block_number=block,
        line_number=line,
        word_number=word_no,
    )


class TestRowGrouping:
    def test_empty_words_returns_empty_layout(self) -> None:
        layout = analyze_layout(())
        assert isinstance(layout, PageLayout)
        assert layout.rows == ()
        assert layout.estimated_columns == ()

    def test_words_on_same_y_form_one_row(self) -> None:
        words = (
            _word("A", x0=10, y0=100),
            _word("B", x0=50, y0=101),
            _word("C", x0=90, y0=99),
        )
        layout = analyze_layout(words, y_tolerance=3.0)
        assert len(layout.rows) == 1
        assert len(layout.rows[0].words) == 3

    def test_words_on_different_y_form_separate_rows(self) -> None:
        words = (
            _word("A", x0=10, y0=100),
            _word("B", x0=10, y0=200),
            _word("C", x0=10, y0=300),
        )
        layout = analyze_layout(words, y_tolerance=3.0)
        assert len(layout.rows) == 3

    def test_rows_are_sorted_top_to_bottom(self) -> None:
        words = (
            _word("Bottom", x0=10, y0=300),
            _word("Top", x0=10, y0=100),
            _word("Middle", x0=10, y0=200),
        )
        layout = analyze_layout(words, y_tolerance=3.0)
        row_texts = [row.text for row in layout.rows]
        assert row_texts == ["Top", "Middle", "Bottom"]

    def test_y_tolerance_controls_row_merging(self) -> None:
        words = (
            _word("A", x0=10, y0=100),
            _word("B", x0=50, y0=106),  # 6pt away
        )
        # Tight tolerance: two separate rows.
        tight = analyze_layout(words, y_tolerance=2.0)
        assert len(tight.rows) == 2

        # Loose tolerance: merged into one row.
        loose = analyze_layout(words, y_tolerance=10.0)
        assert len(loose.rows) == 1


class TestWordOrderingWithinRow:
    def test_words_sorted_by_x_within_row(self) -> None:
        words = (
            _word("Third", x0=300, y0=100),
            _word("First", x0=10, y0=101),
            _word("Second", x0=150, y0=99),
        )
        layout = analyze_layout(words, y_tolerance=3.0)
        assert len(layout.rows) == 1
        ordered_texts = [w.text for w in layout.rows[0].words]
        assert ordered_texts == ["First", "Second", "Third"]

    def test_row_text_property_reflects_x_order(self) -> None:
        words = (
            _word("world", x0=100, y0=0),
            _word("hello", x0=0, y0=0),
        )
        layout = analyze_layout(words, y_tolerance=3.0)
        assert layout.rows[0].text == "hello world"


class TestColumnEstimation:
    def test_recurring_x_positions_detected_as_columns(self) -> None:
        # Two rows, each with words starting at roughly x=10 and x=200 ->
        # two recurring column positions.
        words = (
            _word("R1C1", x0=10, y0=100),
            _word("R1C2", x0=200, y0=100),
            _word("R2C1", x0=11, y0=150),
            _word("R2C2", x0=201, y0=150),
        )
        layout = analyze_layout(words, y_tolerance=3.0, column_tolerance=5.0)
        assert len(layout.rows) == 2
        assert len(layout.estimated_columns) == 2

    def test_one_off_x_position_not_counted_as_column(self) -> None:
        # x=500 only ever appears once across all rows -> not a "column".
        words = (
            _word("R1C1", x0=10, y0=100),
            _word("R2C1", x0=11, y0=150),
            _word("Stray", x0=500, y0=150),
        )
        layout = analyze_layout(words, y_tolerance=3.0, column_tolerance=5.0)
        # x=10/11 recurs (1 column); x=500 appears once (not a column).
        assert len(layout.estimated_columns) == 1

    def test_word_count_property(self) -> None:
        words = (
            _word("A", x0=10, y0=100),
            _word("B", x0=50, y0=100),
            _word("C", x0=10, y0=200),
        )
        layout = analyze_layout(words, y_tolerance=3.0)
        assert layout.word_count == 3
