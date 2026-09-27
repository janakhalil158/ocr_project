# OCR Architecture

Phase 4 turns a preprocessed page image into structured text.
**Baidu Unlimited-OCR is the project's only OCR engine** — Tesseract
and EasyOCR (both previously implemented) and PaddleOCR (previously a
planned-but-unbuilt backend) have been removed. The architecture
remains engine-independent above the engine boundary, mainly so a
config-driven engine name that isn't `"unlimited"` fails with a clear,
controlled error rather than the pipeline hardcoding a class.

```text
                 ┌───────────────┐
                 │ OCR Interface │   src/ocr/base.py  (OCREngine)
                 └───────┬───────┘
                         │
                  ┌──────▼──────┐
                  │Unlimited-OCR│   src/ocr/unlimited_ocr.py
                  │   Engine    │   (baidu/Unlimited-OCR via
                  │ implemented │    Hugging Face transformers)
                  └──────┬──────┘
                         │
                         ▼
     Standardized OCRResult      src/ocr/models.py
   (= PageOCRResult: text, words,
    bounding boxes, confidence,
    engine, processing_time_ms,
    metadata: regions/tables/...)
```

## Selecting an engine

The rest of the pipeline never imports `src/ocr/unlimited_ocr.py`
directly or calls a model API itself. It goes through the factory:

```python
from src.ocr.factory import get_ocr_engine

engine = get_ocr_engine("unlimited")     # the only supported name
engine = get_ocr_engine("tesseract")     # raises OCREngineNotAvailableError —
                                          # removed, not a silent fallback
```

Which engine gets used pipeline-wide is a **configuration** choice, not
a code choice: `OCRConfig.engine` (default `"unlimited"`), settable via
the `OCR_ENGINE` environment variable. `src/main.py` and
`src/ocr/ocr.py` never hardcode an engine. Unlimited-OCR's own knobs
(model name, device, dtype, generation length) are separate
`OCRConfig` fields (`unlimited_model_name`, `unlimited_device`,
`unlimited_dtype`, `unlimited_max_new_tokens`), each overridable via
its own environment variable — see `src/utils/config.py`.

## Running OCR

Two equivalent entry points, both engine-agnostic:

```python
# Via an engine instance directly:
result = engine.process(image)                 # OCREngine.process()

# Via the orchestration module (used by src/main.py):
from src.ocr.ocr import extract_text, extract_text_safe
result = extract_text(image, page_number=1)             # raises OCRError on failure
result = extract_text_safe(image, page_number=1)        # returns None on failure, logs it
```

`image` is a `numpy.ndarray` — the pipeline's standard image type end
to end (PDF rendering, quality assessment, and preprocessing are all
numpy/OpenCV-based already, so OCR follows the same convention rather
than introducing a PIL conversion step nothing else needs).

## OCRResult (`PageOCRResult`)

```text
PageOCRResult  (alias: OCRResult)
├── page_number
├── text                   -- reconstructed, line-structure-preserving
├── language                -- language config actually used (e.g. "ara+eng")
├── engine                  -- engine name ("unlimited_ocr")
├── words: [WordResult]     -- one per detected region (see "Region
│     ├── text                 granularity" below), not one per word
│     ├── confidence           -- always 0.0 for Unlimited-OCR; see
│     └── bounding_box: BoundingBox(x, y, width, height)   "Confidence" below
├── mean_confidence          -- never fabricated; only computed from
│                                words the engine actually reported
├── confidence_level         -- VERY_GOOD / GOOD / MODERATE / POOR --
│                                meaningless for Unlimited-OCR; check
│                                metadata["confidence_available"] first
├── processing_time_ms
└── metadata                 -- see "Region/table metadata" below
```

Bounding boxes are attached per-word (`WordResult.bounding_box`)
rather than kept as a separate parallel list, since a box only means
anything in relation to the word/region it belongs to.

## Engine boundary — what's inside vs. outside `UnlimitedOCREngine`

Everything Unlimited-OCR-specific (loading the Hugging Face model,
parsing its `<|det|>region [x1,y1,x2,y2]<|/det|>content` output,
rescaling its normalized 0-1000 coordinates to real pixels, rendering
table HTML to plain text) lives in `src/ocr/unlimited_ocr.py` and
nowhere else. Everything engine-agnostic — filtering empty/low-
confidence tokens, reconstructing readable multi-line text, computing
mean confidence, bucketing it, timing the call, and folding an
engine's optional region-level detail into `PageOCRResult.metadata` —
lives once in `src/ocr/ocr.py` and applies identically to any future
engine, because it only ever reads the shared structured-data shape
an `OCREngine.recognize_raw()` returns.

## Region granularity

Unlimited-OCR detects whole regions (a title, a paragraph, a table),
not individual words. Each detected region becomes one row in the
Tesseract-shaped `recognize_raw()` output (one `WordResult`, one
"word" in the reconstructed text) rather than a fabricated per-word
split with invented coordinates — an accurate reflection of what the
model actually detected, not a false promise of word-level output.

## Confidence

Unlimited-OCR does not report a confidence score. Every row's `conf`
is the documented placeholder `0.0` — never a fabricated number — and
`recognize_raw()`'s raw dict sets `engine_reports_confidence: False`.
`extract_text()` turns that into
`PageOCRResult.metadata["confidence_available"] = False`. Callers
(the UI included) must check that flag before showing
`mean_confidence`/`confidence_level` as if they meant something.

## Region/table metadata

`recognize_raw()`'s raw dict carries two *optional* parallel arrays
beyond the base Tesseract-shaped contract — `region_type` and
`table_html` — which `extract_text()` recognizes when present and
folds into:

```text
PageOCRResult.metadata = {
    "confidence_available": False,
    "regions": [
        {"type": "title", "text": "...", "bounding_box": {...}},
        {"type": "table", "text": "<plain-text rendering>",
         "bounding_box": {...}, "table_html": "<table>...</table>"},
        ...
    ],
    "tables": [ <the subset of "regions" that have table_html> ],
}
```

A `[Non-Text]` detection is represented as a region with `text: ""`
(type and bounding box preserved) rather than the literal marker
string, so it's excluded from the reconstructed text/word list but
still visible in `metadata["regions"]` for layout purposes. Any
engine that doesn't report `region_type` simply doesn't get
`"regions"`/`"tables"` keys at all — this mechanism is opt-in per
engine, not Unlimited-OCR-specific plumbing bolted onto the shared
layer.

## Bounding box coordinate space

Unlimited-OCR's `<|det|>` boxes are on a normalized 0-1000 grid, not
raw pixels. `src/ocr/unlimited_ocr.py` rescales every box against the
actual rendered-page image's width/height before it reaches
`recognize_raw()`'s return value — everywhere else in the pipeline
(and the UI's bounding-box overlay) can treat `BoundingBox`/
`metadata["regions"][i]["bounding_box"]` as real pixel coordinates
with no further conversion.

## GPU / model loading

The model (6+ GB) is loaded lazily on first use and cached
process-wide, keyed by `(model_name, device, dtype)`
(`src/ocr/unlimited_ocr.py`'s `_MODEL_CACHE`) — constructing an engine
is cheap regardless of how many times `get_ocr_engine()` is called
(once per document in `process_document()`, or once per Streamlit
rerun); the underlying model is only actually loaded once per process.
`ui/app.py` additionally wraps engine construction in
`st.cache_resource` as the idiomatic Streamlit-level layer on top of
that. See `src/ocr/unlimited_ocr.py`'s module docstring for the full
set of documented limitations (confidence, coordinate space, region
granularity).
