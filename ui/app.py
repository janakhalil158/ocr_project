"""
Streamlit UI for the AI Document Processing pipeline.

This file is a PRESENTATION LAYER ONLY. It does not implement PDF
classification, text extraction, quality assessment, preprocessing,
OCR, or layout analysis — all of that still lives in ``src/pdf``,
``src/quality``, ``src/preprocessing``, and ``src/ocr``, exactly as
before. The UI calls a single orchestration function,
``src.pipeline.document_pipeline.process_document``, which itself
just calls those existing modules in the same order
``src/main.py`` does and returns the results as data instead of
printing them.

Run with:

    streamlit run ui/app.py

The CLI (``python -m src.main <pdf_path>``) is unaffected by this
file and keeps working exactly as before.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

from src.pdf.pdf_type_detector import PageType
from src.pipeline.document_pipeline import (
    DocumentPipelineError,
    DocumentProcessingResult,
    PageResult,
    process_document,
)

# --------------------------------------------------------------------------
# Page config / header
# --------------------------------------------------------------------------

st.set_page_config(page_title="AI Document Processing", page_icon="📄", layout="wide")

st.title("AI Document Processing")
st.caption("PDF Classification • OCR • Layout Analysis")

# --------------------------------------------------------------------------
# Small presentation helpers (no pipeline logic lives here)
# --------------------------------------------------------------------------

_PIPELINE_STAGE_LABELS: dict[str, str] = {
    "load": "PDF",
    "classification": "Classification",
    "text_extraction": "Text Extraction",
    "quality_check": "Quality Check",
    "ocr": "OCR",
    "layout_analysis": "Layout Analysis",
}
_PIPELINE_STAGE_ORDER = list(_PIPELINE_STAGE_LABELS.keys())


def _render_stage_indicator(current_stage: str) -> str:
    """Build the 'PDF -> Classification -> ... -> Layout Analysis' banner."""
    current_index = _PIPELINE_STAGE_ORDER.index(current_stage) if current_stage in _PIPELINE_STAGE_ORDER else -1

    parts = []
    for index, stage in enumerate(_PIPELINE_STAGE_ORDER):
        label = _PIPELINE_STAGE_LABELS[stage]
        if index < current_index:
            parts.append(f"✅ {label}")
        elif index == current_index:
            parts.append(f"🔵 **{label}**")
        else:
            parts.append(f"⚪ {label}")
    return "  →  ".join(parts)


def _draw_bounding_boxes(image: np.ndarray, words, max_boxes: int = 500) -> Image.Image:
    """Draw OCR word/region bounding boxes on a copy of a page image."""
    pil_image = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(pil_image)
    for word in list(words)[:max_boxes]:
        box = word.bounding_box
        x0, y0 = box.x, box.y
        x1, y1 = box.x + box.width, box.y + box.height
        draw.rectangle([x0, y0, x1, y1], outline=(220, 38, 38), width=2)
    return pil_image


def _save_uploaded_pdf(uploaded_file) -> Path:
    """Persist the uploaded file to a temp path the pipeline can open by path."""
    tmp_dir = Path(tempfile.gettempdir()) / "ai_document_processing_ui"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / uploaded_file.name
    tmp_path.write_bytes(uploaded_file.getvalue())
    return tmp_path


def _format_bytes(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------

if "result" not in st.session_state:
    st.session_state.result = None
if "result_error" not in st.session_state:
    st.session_state.result_error = None
if "pdf_path" not in st.session_state:
    st.session_state.pdf_path = None

# --------------------------------------------------------------------------
# Upload section
# --------------------------------------------------------------------------

st.subheader("Upload")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    if st.session_state.pdf_path is None or Path(st.session_state.pdf_path).name != uploaded_file.name:
        # New file selected: reset any previous run's results.
        st.session_state.pdf_path = str(_save_uploaded_pdf(uploaded_file))
        st.session_state.result = None
        st.session_state.result_error = None

    col1, col2, col3 = st.columns(3)
    col1.metric("Filename", uploaded_file.name)
    col2.metric("File size", _format_bytes(uploaded_file.size))

    try:
        from src.pdf.loader import get_pdf_metadata

        preview_metadata = get_pdf_metadata(st.session_state.pdf_path)
        col3.metric("Pages", preview_metadata.num_pages)
    except Exception:
        col3.metric("Pages", "—")

    st.divider()
    st.subheader("Processing")

    run_clicked = st.button("Start processing", type="primary")

    if run_clicked:
        stage_banner = st.empty()
        progress_bar = st.progress(0.0)
        log_box = st.empty()
        log_lines: list[str] = []

        def _on_progress(stage: str, message: str) -> None:
            stage_banner.markdown(_render_stage_indicator(stage))
            if stage in _PIPELINE_STAGE_ORDER:
                progress_bar.progress((_PIPELINE_STAGE_ORDER.index(stage) + 1) / len(_PIPELINE_STAGE_ORDER))
            log_lines.append(f"- {message}")
            log_box.markdown("\n".join(log_lines[-8:]))

        try:
            with st.spinner("Running pipeline..."):
                result = process_document(st.session_state.pdf_path, progress_callback=_on_progress)
            st.session_state.result = result
            st.session_state.result_error = None
            progress_bar.progress(1.0)
            st.success(f"Processing complete in {result.total_processing_time_ms:.0f} ms.")
        except DocumentPipelineError as exc:
            st.session_state.result = None
            st.session_state.result_error = str(exc)
        except Exception as exc:  # last-resort catch so the UI always shows *something*
            st.session_state.result = None
            st.session_state.result_error = f"Unexpected error: {exc}"

if st.session_state.result_error:
    st.error(f"Processing failed: {st.session_state.result_error}")

# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------

result: DocumentProcessingResult | None = st.session_state.result

if result is not None:
    st.divider()
    st.subheader("Results")

    if result.ocr_engine_error:
        st.warning(
            f"OCR engine '{result.ocr_engine_name}' is not available: {result.ocr_engine_error}\n\n"
            "Pages that needed OCR could not be recognized (see the OCR tab)."
        )

    tab_overview, tab_classification, tab_text, tab_ocr, tab_layout = st.tabs(
        ["Overview", "Page Classification", "Extracted Text", "OCR", "Layout"]
    )

    # --- Overview ---
    with tab_overview:
        c1, c2, c3 = st.columns(3)
        c1.metric("Filename", result.metadata.filename)
        c1.metric("Pages", result.metadata.num_pages)
        c2.metric("Classification", result.classification.document_type.value)
        c2.metric("Processing time", f"{result.total_processing_time_ms:.0f} ms")
        c3.metric("OCR engine", result.ocr_engine_name.upper())
        c3.metric("OCR language", result.ocr_language)

    # --- Page Classification ---
    with tab_classification:
        rows = [
            {"Page": p.page_number, "Type": p.page_type.value, "Characters": p.character_count}
            for p in result.pages
        ]
        st.table(rows)

    # Page selector shared by the text-heavy tabs below.
    page_numbers = [p.page_number for p in result.pages]
    page_lookup: dict[int, PageResult] = {p.page_number: p for p in result.pages}

    # --- Extracted Text ---
    with tab_text:
        selected_page = st.selectbox("Page", page_numbers, key="text_page_select")
        page = page_lookup[selected_page]

        if page.error:
            st.error(page.error)

        if page.page_type is PageType.TEXT:
            if page.native_quality_good is False:
                st.warning("Native extraction unreliable → EasyOCR fallback")
            elif page.native_quality_good is True:
                st.success("Native extraction quality: GOOD")

        if page.native_text:
            st.text_area("Native extracted text", page.native_text, height=300)

        if page.used_ocr and page.ocr_result is not None:
            st.text_area("OCR extracted text", page.ocr_result.text or "(no text recognized)", height=300)
        elif page.used_ocr and page.ocr_error:
            st.error(f"OCR error: {page.ocr_error}")

    # --- OCR ---
    with tab_ocr:
        selected_page_ocr = st.selectbox("Page", page_numbers, key="ocr_page_select")
        page = page_lookup[selected_page_ocr]

        if not page.used_ocr:
            st.info("This page used native PDF text; OCR did not run.")
        elif page.ocr_error:
            st.error(f"OCR failed for this page: {page.ocr_error}")
        elif page.ocr_result is not None:
            ocr = page.ocr_result
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Average confidence", f"{ocr.mean_confidence:.1f}%")
            m2.metric("Confidence level", ocr.confidence_level.value)
            m3.metric("Detected regions", len(ocr.words))
            m4.metric("Processing time", f"{ocr.processing_time_ms:.0f} ms" if ocr.processing_time_ms else "—")

            st.text_area("OCR text", ocr.text or "(no text recognized)", height=250)

            display_image = page.processed_image if page.processed_image is not None else page.rendered_image
            if display_image is not None and ocr.words:
                st.image(
                    _draw_bounding_boxes(display_image, ocr.words),
                    caption=f"Detected text regions (page {page.page_number})",
                    use_container_width=True,
                )
            elif display_image is not None:
                st.image(display_image, caption=f"Page {page.page_number}", use_container_width=True)

    # --- Layout ---
    with tab_layout:
        selected_page_layout = st.selectbox("Page", page_numbers, key="layout_page_select")
        page = page_lookup[selected_page_layout]

        if page.layout is not None:
            layout = page.layout
            l1, l2, l3 = st.columns(3)
            l1.metric("Rows detected", len(layout.rows))
            l2.metric("Approx. columns", len(layout.estimated_columns))
            l3.metric("Words", layout.word_count)

            row_data = [
                {"Row": i + 1, "Text": row.text}
                for i, row in enumerate(layout.rows)
            ]
            if row_data:
                st.dataframe(row_data, use_container_width=True, height=350)
            else:
                st.info("No rows reconstructed for this page.")

        elif page.used_ocr and page.ocr_result is not None and page.ocr_result.words:
            st.caption("This page went through OCR rather than native-text layout reconstruction; "
                       "showing detected text regions instead.")
            m1, m2 = st.columns(2)
            m1.metric("Detected regions", len(page.ocr_result.words))
            m2.metric("Average confidence", f"{page.ocr_result.mean_confidence:.1f}%")

            display_image = page.processed_image if page.processed_image is not None else page.rendered_image
            if display_image is not None:
                st.image(
                    _draw_bounding_boxes(display_image, page.ocr_result.words),
                    caption=f"Detected text regions (page {page.page_number})",
                    use_container_width=True,
                )
        else:
            st.info("No layout information available for this page.")

elif uploaded_file is None:
    st.info("Upload a PDF above to get started.")