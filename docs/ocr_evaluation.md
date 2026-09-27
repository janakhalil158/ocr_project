# OCR Evaluation & Benchmarking

Phase 5 measures OCR quality objectively against ground truth, rather
than trusting an engine's own confidence score. It was originally
built to compare Tesseract against a future second engine; Baidu
Unlimited-OCR is now the project's only engine (see
`docs/ocr_architecture.md`), so today it's a straightforward
accuracy/regression benchmark rather than a cross-engine comparison —
the tooling and metrics below are unchanged either way.

## Why OCR confidence alone is not enough

Every `PageOCRResult` already carries a `mean_confidence` from the
engine. That number answers "how sure was the engine?" — not "was the
engine right?". An engine can misread a word with high confidence, and
a confidently wrong answer looks identical to a confidently correct one
from the confidence score alone.

Worse, confidence is not comparable *across engines*: Tesseract's 0–100
scale and PaddleOCR's score are produced by different models with
different calibration. Comparing them directly would be meaningless.

CER and WER compare output against **verified ground truth**, which is
engine-independent and therefore the only sound basis for a
Tesseract-vs-PaddleOCR claim. Confidence is still recorded, but as a
descriptive signal alongside the real metrics — not as the measure of
accuracy.

## CER — Character Error Rate

```text
CER = levenshtein_distance(reference, hypothesis) / len(reference)
```

Character-level edit distance (substitutions, insertions, deletions,
each weighted 1) divided by reference length. `0.0` is perfect.

**Empty-reference behaviour** is defined rather than left to divide by
zero: empty reference + empty hypothesis → `0.0` (nothing expected,
nothing produced); empty reference + non-empty hypothesis → `1.0`
(everything produced is spurious). `1.0` keeps the value on the same
scale as every other sample so it can be averaged safely.

CER is **not capped at 1.0** in the normal case. An engine that
hallucinates far more text than the reference contains can legitimately
exceed 1.0, and clamping it would hide a real failure.

## WER — Word Error Rate

```text
WER = levenshtein_distance(reference_words, hypothesis_words) / len(reference_words)
```

The same edit distance applied to word-token lists instead of
characters, with the same normalization and the same empty-reference
convention. WER is typically higher than CER, since one wrong character
invalidates a whole word.

## Normalization — exactly what is done

Applied to **both** reference and hypothesis before comparison
(`src/ocr/metrics.py::normalize_text`):

1. Unicode **NFC** normalization — so visually identical text using
   decomposed code points (common with Arabic diacritics and accented
   Latin) compares equal instead of registering as errors.
2. Line breaks (`\n`, `\r\n`, `\r`) collapsed to single spaces — *where*
   the engine broke lines is a separate concern from character accuracy.
3. Runs of whitespace collapsed to one space.
4. Leading/trailing whitespace stripped.

Deliberately **NOT** done, because each would flatter the engine and
corrupt the measurement:

* No lowercasing — case errors are real OCR errors.
* No punctuation stripping.
* No spell correction or dictionary snapping.
* **No Arabic-specific rewriting**: no tashkeel/diacritic removal, no
  alef/hamza unification, no tatweel stripping, no
  Arabic-Indic→ASCII digit mapping. These destroy real character
  distinctions and would silently inflate Arabic scores.

Arabic text passes through with its characters intact.

## Dataset format

```text
data/evaluation/
├── dataset.json
├── images/
├── ground_truth/
└── results/
```

`dataset.json`:

```json
{
  "samples": [
    {
      "id": "english_clean_001",
      "image": "images/english_clean_001.png",
      "ground_truth": "ground_truth/english_clean_001.txt",
      "language": "eng",
      "category": "clean",
      "notes": "synthetic; ground truth known by construction"
    }
  ]
}
```

Only `id`, `image`, and `ground_truth` are required. Categories are
free-form strings — `clean`, `blurry`, `skewed`, `low_contrast`,
`phone_photo`, `arabic`, `english`, `mixed_language` — and new ones can
be added without code changes. `data/sample/` is untouched and separate.

## Ground truth

**Ground truth must never be produced by running OCR.** Scoring OCR
output against OCR output is circular and would report 0.0 error
regardless of actual quality.

Two legitimate sources:

1. **Known by construction** — the current synthetic samples, rendered
   from source text by `scripts/generate_evaluation_samples.py`. The
   text is defined first and rendered to an image, so the truth is
   exact.
2. **Human transcription** — for real documents (the certificate PDF,
   scanned pages). A person reads the image and types what is actually
   there, including its errors.

To add a real-document sample: render the PDF page to PNG (the pipeline
uses `src/pdf/page_renderer.py`), drop it in `images/`, hand-transcribe
the text into `ground_truth/<id>.txt`, and add the manifest entry.
Evaluation works on **images, not PDFs** — PDF handling stays in the
PDF layer.

## Evaluation workflow

```text
Evaluation Dataset
       │
       ├── Image
       └── Ground Truth
              │
              ▼
        OCR Evaluator          src/ocr/evaluation.py
              │
              ▼
        OCREngine              src/ocr/base.py
              │
       ┌──────┴──────┐
       ▼             ▼
   Tesseract     PaddleOCR
  (removed)      (never built)
       │             │
       └──────┬──────┘
              ▼
       OCR Prediction
              │
       ┌──────┴──────┐
       ▼             ▼
      CER           WER        src/ocr/metrics.py
       │             │
       └──────┬──────┘
              ▼
       Benchmark Results       data/evaluation/results/
```

The evaluator calls `engine.process(image)` through the `OCREngine`
interface and contains **no engine-specific logic** — identical
images, identical ground truth, identical normalization, identical
metrics, whichever engine is configured. Baidu Unlimited-OCR
(`src/ocr/unlimited_ocr.py`) is now the project's only engine; the
diagram above and the historical notes below describe this module's
original two-engine design intent, kept for context.

## Running an evaluation

```bash
# Generate the synthetic sample set (first time only)
.venv/bin/python scripts/generate_evaluation_samples.py

# Run the benchmark with the configured engine
.venv/bin/python -m src.ocr.evaluation

# Explicit engine / dataset / output location
.venv/bin/python -m src.ocr.evaluation --engine unlimited \
    --dataset data/evaluation/dataset.json \
    --results-dir data/evaluation/results

# Print the summary without writing files
.venv/bin/python -m src.ocr.evaluation --no-save
```

## Result format

Written to `data/evaluation/results/` as both `results.json` (summary +
per-sample detail) and `results.csv` (flat, one row per sample).

Per sample: `dataset_id`, `engine`, `language`, `category`,
`reference_text_length`, `recognized_text_length`,
`reference_word_count`, `recognized_word_count`, `cer`, `wer`,
`ocr_confidence`, `processing_time_ms`, `success`, `error`.

A failed sample has `success: false`, an `error` message, and **`cer`/
`wer` left as `null`** — never a placeholder `0.0`. Failures are counted
separately in the summary so a crashed sample can never masquerade as a
perfect score.

Summary: sample/success/failure counts, average and median CER and WER,
average confidence, average processing time, fastest and slowest
samples, plus the same statistics broken down per category.

## Unlimited-OCR as the only engine

Running the benchmark records Baidu Unlimited-OCR's numbers on a fixed
dataset — there is no second engine to compare against (Tesseract and
EasyOCR have been removed; PaddleOCR was never built). Unlimited-OCR
does not report a confidence score (see `src/ocr/unlimited_ocr.py`),
so `ocr_confidence` in the result format below is not meaningful for
current runs — CER/WER against ground truth are the metrics that
matter here.

## Adding another engine later

No changes to this evaluation layer would be required. Implement
`SomeOtherEngine(OCREngine)`, register it in `src/ocr/factory.py`
(which would then need to support more than one engine name again),
then:

```bash
.venv/bin/python -m src.ocr.evaluation --engine unlimited --results-dir data/evaluation/results/unlimited
.venv/bin/python -m src.ocr.evaluation --engine some_other_engine --results-dir data/evaluation/results/some_other_engine
```

## A caution learned while building this

The first Arabic benchmark run reported CER 0.83 — apparently terrible
Arabic accuracy. It was not an engine problem. The sample *generator*
was applying bidi reordering on top of Pillow's own RAQM layout, which
rendered Arabic lines character-reversed; Tesseract read the reversed
image correctly and scored as if it had failed. After the fix, the same
sample scores CER 0.0.

The lesson is general: **a bad benchmark number is a claim about your
harness until you have ruled the harness out.** Before reporting any
poor score as an engine weakness, inspect the actual OCR output next to
the ground truth. `scripts/generate_evaluation_samples.py` documents
the RAQM dependency in `_shape_arabic()`; check
`PIL.features.check("raqm")` if you regenerate samples elsewhere.
