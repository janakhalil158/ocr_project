# OCR Architecture

Phase 4 turns a preprocessed page image into structured text. The
architecture is deliberately engine-independent: nothing outside
`src/ocr/` knows or cares whether Tesseract or some future engine
actually did the recognition.

```text
                 ┌───────────────┐
                 │ OCR Interface │   src/ocr/base.py  (OCREngine)
                 └───────┬───────┘
                         │
              ┌──────────┴──────────┐
              │                     │
       ┌──────▼──────┐       ┌──────▼──────┐
       │  Tesseract  │       │  PaddleOCR  │
       │   Engine    │       │   (future)  │
       │ implemented │       │  not built  │
       └──────┬──────┘       └─────────────┘
              │
              ▼
     Standardized OCRResult      src/ocr/models.py
   (= PageOCRResult: text, words,
    bounding boxes, confidence,
    engine, processing_time_ms)
```

## Selecting an engine

The rest of the pipeline never imports an engine-specific module or
calls `pytesseract` directly. It goes through the factory:

```python
from src.ocr.factory import get_ocr_engine

engine = get_ocr_engine("tesseract")     # works today
engine = get_ocr_engine("paddleocr")     # raises OCREngineNotAvailableError
                                          # with a clear explanation, until implemented
```

Which engine gets used pipeline-wide is a **configuration** choice, not
a code choice: `OCRConfig.engine` (default `"tesseract"`), settable via
the `OCR_ENGINE` environment variable. `src/main.py` and
`src/ocr/ocr.py` never hardcode an engine — changing

```python
OCR_ENGINE=tesseract
```

to

```python
OCR_ENGINE=paddleocr
```

will switch the whole pipeline's OCR engine once a `PaddleOCREngine`
is added to the factory, with no other code touched.

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
├── engine                  -- engine name (e.g. "tesseract")
├── words: [WordResult]
│     ├── text
│     ├── confidence
│     └── bounding_box: BoundingBox(x, y, width, height)
├── mean_confidence          -- never fabricated; only computed from
│                                words the engine actually reported
├── confidence_level         -- VERY_GOOD / GOOD / MODERATE / POOR
├── processing_time_ms
└── metadata                 -- engine-specific extras (e.g. PSM/OEM)
```

Bounding boxes are attached per-word (`WordResult.bounding_box`)
rather than kept as a separate parallel list, since a box only means
anything in relation to the word it belongs to.

## Engine boundary — what's inside vs. outside `TesseractOCREngine`

Everything Tesseract-specific (the `pytesseract` calls, PSM/OEM,
`TesseractNotFoundError`/`TesseractError` handling, the executable
path override) lives in `src/ocr/tesseract_ocr.py` and nowhere else.
Everything engine-agnostic — filtering empty/low-confidence tokens,
reconstructing readable multi-line text, computing mean confidence,
bucketing it, timing the call — lives once in `src/ocr/ocr.py` and
applies identically to any future engine, because it only ever reads
the shared structured-data shape an `OCREngine.recognize_raw()`
returns.

## PaddleOCR status

Not installed, not implemented, not required by anything in this
environment. `src/ocr/base.py` defines the interface a
`PaddleOCREngine` would need to implement; `src/ocr/factory.py`
already recognizes `"paddleocr"` as a real, planned engine name and
fails with a clear, actionable error rather than an import error —
so requesting it today is a controlled failure, not a crash.
