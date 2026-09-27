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
printing them. There is no separate OCR implementation here — the UI
and the CLI (``python -m src.main <pdf_path>``) run the exact same
pipeline.

Run with:

    streamlit run ui/app.py
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from src.utils.config import OCR_CONFIG

# --------------------------------------------------------------------------
# Page config / header
# --------------------------------------------------------------------------

st.set_page_config(page_title="AI Document Processor", page_icon="📄", layout="wide")


def _detect_gpu_status() -> Tuple[bool, str]:
    """
    Report whether a CUDA GPU is available, without ever surfacing a raw
    traceback to the user — Unlimited-OCR requires GPU execution (see
    src/ocr/unlimited_ocr.py), so this is shown prominently rather than
    only discovered when processing fails.
    """
    try:
        import torch  # type: ignore
    except ImportError:
        return False, "PyTorch is not installed — Unlimited-OCR requires GPU execution."

    try:
        if torch.cuda.is_available():
            return True, torch.cuda.get_device_name(0)
    except Exception:
        pass
    return False, "No CUDA-capable GPU detected — Unlimited-OCR requires GPU execution."


st.title("AI Document Processor")
st.caption("Intelligent document extraction powered by Baidu Unlimited-OCR")

_gpu_available, _gpu_label = _detect_gpu_status()
if _gpu_available:
    st.caption(f"🟢 GPU: {_gpu_label}")
else:
    st.warning(f"⚠️ {_gpu_label} Processing will fail until a GPU is available.")

# --------------------------------------------------------------------------
# Small presentation helpers (no pipeline logic lives here)
# --------------------------------------------------------------------------

_PIPELINE_STAGE_LABELS: Dict[str, str] = {
    "load": "PDF",
    "classification": "Classification",
    "text_extraction": "Text Extraction",
    "quality_check": "Quality Check",
    "ocr": "OCR",
    "layout_analysis": "Layout Analysis",
}
_PIPELINE_STAGE_ORDER = list(_PIPELINE_STAGE_LABELS.keys())
_PAGE_NUM_IN_MESSAGE = re.compile(r"page (\d+)", re.IGNORECASE)


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


def _render_page_checklist(page_status: Dict[int, str]) -> str:
    """
    Render a per-page '✓ Page 1  ⟳ Page 2  ○ Page 3' checklist from
    whatever page numbers have actually appeared in progress messages so
    far — reflects real callback data rather than a fabricated progress
    animation (no total-page-count is known inside the callback until
    processing finishes).
    """
    icons = {"done": "✓", "running": "⟳", "pending": "○"}
    return "&nbsp;&nbsp;".join(
        f"{icons.get(status, '○')} Page {page_number}" for page_number, status in sorted(page_status.items())
    )


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


def _table_text_to_records(text: str) -> List[Dict[str, str]]:
    """Turn a table's tab/newline-separated plain text (see
    src/ocr/unlimited_ocr.py's html_table_to_text) into row dicts a
    Streamlit dataframe can render with real headers."""
    rows = [line.split("\t") for line in text.split("\n") if line.strip()]
    if not rows:
        return []
    header, *body = rows
    if not body:
        return [{f"Col {i + 1}": cell for i, cell in enumerate(header)}]
    return [{header[i] if i < len(header) else f"Col {i + 1}": cell for i, cell in enumerate(row)} for row in body]


def _collect_all_tables(result: DocumentProcessingResult) -> List[Dict[str, Any]]:
    """Every detected table across every page, with its page number attached."""
    tables: List[Dict[str, Any]] = []
    for page in result.pages:
        if page.ocr_result is None:
            continue
        for table in page.ocr_result.metadata.get("tables", []):
            tables.append({"page": page.page_number, **table})
    return tables


def _build_combined_text(result: DocumentProcessingResult) -> str:
    """The best available text for every page, concatenated for a plain .txt export."""
    parts: List[str] = []
    for page in result.pages:
        parts.append(f"=== Page {page.page_number} ===")
        if page.native_text and page.native_quality_good:
            parts.append(page.native_text)
        elif page.ocr_result is not None:
            parts.append(page.ocr_result.text or "(no text recognized)")
        elif page.native_text:
            parts.append(page.native_text)
        else:
            parts.append("(no text extracted)")
        parts.append("")
    return "\n".join(parts)


def _build_export_payload(result: DocumentProcessingResult) -> Dict[str, Any]:
    """
    Structured JSON export: document info, per-page text/OCR/layout
    detail. Deliberately built from plain data only (strings, numbers,
    dicts already produced by PageOCRResult.to_dict()) — never a raw
    numpy image array or any internal model object.
    """
    return {
        "filename": result.metadata.filename,
        "file_size_bytes": result.metadata.file_size_bytes,
        "num_pages": result.metadata.num_pages,
        "classification": result.classification.document_type.value,
        "ocr_engine": result.ocr_engine_name,
        "ocr_language": result.ocr_language,
        "ocr_engine_error": result.ocr_engine_error,
        "processing_time_ms": result.total_processing_time_ms,
        "pages": [
            {
                "page_number": page.page_number,
                "page_type": page.page_type.value,
                "character_count": page.character_count,
                "native_text": page.native_text,
                "native_quality_good": page.native_quality_good,
                "used_ocr": page.used_ocr,
                "error": page.error,
                "ocr_error": page.ocr_error,
                "ocr_result": page.ocr_result.to_dict() if page.ocr_result is not None else None,
                "layout": (
                    {
                        "row_count": len(page.layout.rows),
                        "estimated_columns": len(page.layout.estimated_columns),
                        "rows": [row.text for row in page.layout.rows],
                    }
                    if page.layout is not None
                    else None
                ),
            }
            for page in result.pages
        ],
    }


@st.cache_resource(show_spinner=False)
def _get_cached_ocr_engine():
    """
    Build (and Streamlit-cache) the configured OCR engine once per
    session/process. UnlimitedOCREngine itself also caches the
    underlying Hugging Face model process-wide the first time it's
    actually used (see src/ocr/unlimited_ocr.py's _MODEL_CACHE), so the
    6+ GB model is loaded at most once no matter how many documents are
    processed in this session — this is the idiomatic Streamlit-level
    layer on top of that.
    """
    from src.ocr.factory import get_ocr_engine

    return get_ocr_engine()


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------

if "result" not in st.session_state:
    st.session_state.result = None
if "result_error" not in st.session_state:
    st.session_state.result_error = None
if "result_error_detail" not in st.session_state:
    st.session_state.result_error_detail = None
if "pdf_path" not in st.session_state:
    st.session_state.pdf_path = None

# --------------------------------------------------------------------------
# Upload section
# --------------------------------------------------------------------------

st.subheader("Upload your document")

uploaded_file = st.file_uploader("Drag & drop a PDF here, or browse files", type=["pdf"])

if uploaded_file is not None:
    if st.session_state.pdf_path is None or Path(st.session_state.pdf_path).name != uploaded_file.name:
        # New file selected: reset any previous run's results.
        st.session_state.pdf_path = str(_save_uploaded_pdf(uploaded_file))
        st.session_state.result = None
        st.session_state.result_error = None
        st.session_state.result_error_detail = None

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Document", uploaded_file.name)
    col2.metric("File size", _format_bytes(uploaded_file.size))

    try:
        from src.pdf.loader import get_pdf_metadata
        from src.pdf.pdf_type_detector import detect_pdf_type

        preview_metadata = get_pdf_metadata(st.session_state.pdf_path)
        col3.metric("Pages", preview_metadata.num_pages)
        preview_classification = detect_pdf_type(st.session_state.pdf_path)
        col4.metric("Type", preview_classification.document_type.value)
    except Exception:
        col3.metric("Pages", "—")
        col4.metric("Type", "—")

    st.divider()
    st.subheader("Process Document")

    run_clicked = st.button("Process Document", type="primary")

    if run_clicked:
        if not _gpu_available:
            st.error(
                "Unlimited-OCR requires a GPU, and none was detected in this environment. "
                "Processing was not started."
            )
        else:
            stage_banner = st.empty()
            progress_bar = st.progress(0.0)
            page_checklist_box = st.empty()
            log_box = st.empty()

            log_lines: List[str] = []
            page_status: Dict[int, str] = {}

            def _on_progress(stage: str, message: str) -> None:
                stage_banner.markdown(_render_stage_indicator(stage))
                if stage in _PIPELINE_STAGE_ORDER:
                    progress_bar.progress((_PIPELINE_STAGE_ORDER.index(stage) + 1) / len(_PIPELINE_STAGE_ORDER))

                match = _PAGE_NUM_IN_MESSAGE.search(message)
                if match:
                    page_number = int(match.group(1))
                    for number, status in page_status.items():
                        if status == "running":
                            page_status[number] = "done"
                    page_status[page_number] = "running"
                    page_checklist_box.markdown(_render_page_checklist(page_status), unsafe_allow_html=True)

                log_lines.append(f"- {message}")
                log_box.markdown("\n".join(log_lines[-8:]))

            try:
                with st.spinner("Loading Unlimited-OCR model (first run only)..."):
                    _get_cached_ocr_engine()

                with st.spinner("Processing document..."):
                    result = process_document(st.session_state.pdf_path, progress_callback=_on_progress)

                for number in page_status:
                    page_status[number] = "done"
                page_checklist_box.markdown(_render_page_checklist(page_status), unsafe_allow_html=True)

                st.session_state.result = result
                st.session_state.result_error = None
                st.session_state.result_error_detail = None
                progress_bar.progress(1.0)
                st.success(f"Processing complete in {result.total_processing_time_ms:.0f} ms.")
            except DocumentPipelineError as exc:
                st.session_state.result = None
                st.session_state.result_error = str(exc)
                st.session_state.result_error_detail = None
            except Exception as exc:  # last-resort catch so the UI always shows *something* clean
                st.session_state.result = None
                st.session_state.result_error = (
                    "Something went wrong while processing this document. "
                    "Please check the document and try again."
                )
                st.session_state.result_error_detail = repr(exc)

if st.session_state.result_error:
    st.error(f"Processing failed: {st.session_state.result_error}")
    if st.session_state.result_error_detail:
        with st.expander("Technical details"):
            st.code(st.session_state.result_error_detail)

# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------

result: Optional[DocumentProcessingResult] = st.session_state.result

if result is not None:
    st.divider()
    st.subheader("Results")

    if result.ocr_engine_error:
        st.warning(
            f"OCR engine '{result.ocr_engine_name}' is not available: {result.ocr_engine_error}\n\n"
            "Pages that needed OCR could not be recognized (see the Extracted Text tab)."
        )

    export_col1, export_col2, export_col3 = st.columns(3)
    export_col1.download_button(
        "⬇ Download Text (.txt)",
        data=_build_combined_text(result),
        file_name=f"{Path(result.metadata.filename).stem}.txt",
        mime="text/plain",
        use_container_width=True,
    )
    export_col2.download_button(
        "⬇ Download Structured JSON",
        data=json.dumps(_build_export_payload(result), ensure_ascii=False, indent=2),
        file_name=f"{Path(result.metadata.filename).stem}.json",
        mime="application/json",
        use_container_width=True,
    )
    all_tables = _collect_all_tables(result)
    export_col3.download_button(
        "⬇ Download Tables (JSON)",
        data=json.dumps(all_tables, ensure_ascii=False, indent=2),
        file_name=f"{Path(result.metadata.filename).stem}_tables.json",
        mime="application/json",
        disabled=not all_tables,
        use_container_width=True,
    )

    tab_overview, tab_classification, tab_text, tab_tables, tab_regions, tab_metadata = st.tabs(
        ["Overview", "Page Classification", "Extracted Text", "Tables", "Layout / Regions", "Metadata"]
    )

    # --- Overview ---
    with tab_overview:
        c1, c2, c3 = st.columns(3)
        c1.metric("Filename", result.metadata.filename)
        c1.metric("Pages", result.metadata.num_pages)
        c2.metric("Document type", result.classification.document_type.value)
        c2.metric("Processing time", f"{result.total_processing_time_ms:.0f} ms")
        c3.metric("OCR engine", "Unlimited-OCR" if result.ocr_engine_name == "unlimited" else result.ocr_engine_name)
        c3.metric("GPU", _gpu_label if _gpu_available else "Not available")
        st.caption(
            "Unlimited-OCR does not report a per-page confidence score, so no confidence "
            "percentage is shown for OCR'd pages — see the Extracted Text tab for a note on this "
            "per page."
        )

    # --- Page Classification ---
    with tab_classification:
        rows = [
            {"Page": p.page_number, "Type": p.page_type.value, "Characters": p.character_count}
            for p in result.pages
        ]
        st.table(rows)

    # Page selector shared by the per-page tabs below.
    page_numbers = [p.page_number for p in result.pages]
    page_lookup: Dict[int, PageResult] = {p.page_number: p for p in result.pages}

    # --- Extracted Text ---
    with tab_text:
        selected_page = st.selectbox("Page", page_numbers, key="text_page_select")
        page = page_lookup[selected_page]

        if page.error:
            st.error(page.error)

        if page.page_type is PageType.TEXT:
            if page.native_quality_good is False:
                st.warning("Native extraction unreliable → OCR fallback")
            elif page.native_quality_good is True:
                st.success("Native extraction quality: GOOD")

        if page.native_text:
            st.text_area("Native extracted text", page.native_text, height=280)

        if page.used_ocr and page.ocr_result is not None:
            ocr = page.ocr_result
            if not ocr.metadata.get("confidence_available", True):
                st.caption("ℹ️ Unlimited-OCR does not report a confidence score for this text.")
            st.text_area("OCR extracted text", ocr.text or "(no text recognized)", height=280)
        elif page.used_ocr and page.ocr_error:
            st.error(f"OCR error: {page.ocr_error}")

    # --- Tables ---
    with tab_tables:
        if not all_tables:
            st.info("No tables were detected in this document.")
        else:
            for i, table in enumerate(all_tables, start=1):
                st.markdown(f"**Table {i} — page {table['page']}**")
                records = _table_text_to_records(table.get("text", ""))
                if records:
                    st.dataframe(records, use_container_width=True)
                else:
                    st.caption("(empty table)")
                if table.get("table_html"):
                    with st.expander("Raw HTML"):
                        st.code(table["table_html"], language="html")
                st.divider()

    # --- Layout / Regions ---
    with tab_regions:
        selected_page_regions = st.selectbox("Page", page_numbers, key="regions_page_select")
        page = page_lookup[selected_page_regions]

        if page.layout is not None:
            layout = page.layout
            l1, l2, l3 = st.columns(3)
            l1.metric("Rows detected", len(layout.rows))
            l2.metric("Approx. columns", len(layout.estimated_columns))
            l3.metric("Words", layout.word_count)

            row_data = [{"Row": i + 1, "Text": row.text} for i, row in enumerate(layout.rows)]
            if row_data:
                st.dataframe(row_data, use_container_width=True, height=350)
            else:
                st.info("No rows reconstructed for this page.")

        elif page.used_ocr and page.ocr_result is not None:
            ocr = page.ocr_result
            regions = ocr.metadata.get("regions")

            if regions:
                m1, m2 = st.columns(2)
                m1.metric("Detected regions", len(regions))
                m2.metric(
                    "Region types",
                    len({r["type"] for r in regions}),
                )
                region_rows = [
                    {
                        "Type": r["type"],
                        "Text": (r["text"][:120] + "…") if len(r["text"]) > 120 else r["text"] or "(no text)",
                        "Page X": r["bounding_box"]["x"],
                        "Page Y": r["bounding_box"]["y"],
                    }
                    for r in regions
                ]
                st.dataframe(region_rows, use_container_width=True, height=300)
            elif ocr.words:
                st.caption("Showing detected text regions for this page.")
                m1, m2 = st.columns(2)
                m1.metric("Detected regions", len(ocr.words))
                m2.metric("Words", len(ocr.words))

            display_image = page.processed_image if page.processed_image is not None else page.rendered_image
            if display_image is not None and ocr.words:
                try:
                    st.image(
                        _draw_bounding_boxes(display_image, ocr.words),
                        caption=f"Detected regions (page {page.page_number})",
                        use_container_width=True,
                    )
                except Exception:
                    st.caption("Could not render the page preview with bounding boxes for this page.")
            elif display_image is not None:
                st.image(display_image, caption=f"Page {page.page_number}", use_container_width=True)
        else:
            st.info("No layout or region information available for this page.")

    # --- Metadata ---
    with tab_metadata:
        selected_page_meta = st.selectbox("Page", page_numbers, key="metadata_page_select")
        page = page_lookup[selected_page_meta]

        st.caption("Structured technical detail for this page, for inspection/debugging.")
        page_payload = {
            "page_number": page.page_number,
            "page_type": page.page_type.value,
            "character_count": page.character_count,
            "native_quality_good": page.native_quality_good,
            "used_ocr": page.used_ocr,
            "error": page.error,
            "ocr_error": page.ocr_error,
            "ocr_result": page.ocr_result.to_dict() if page.ocr_result is not None else None,
        }
        st.json(page_payload)

elif uploaded_file is None:
    st.info("Upload a PDF above to get started.")
