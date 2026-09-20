"""
Phase 5: OCR evaluation & benchmarking.

Measures OCR quality objectively against verified ground truth, so
Tesseract can be established as a baseline before any second engine is
introduced. Deliberately engine-agnostic: the evaluator only talks to
the :class:`~src.ocr.base.OCREngine` abstraction, so the identical
images, ground truth, normalization, and metrics can later score
PaddleOCR with no changes here.

    Evaluation Dataset
           |
           +-- Image
           +-- Ground Truth
                  |
                  v
            OCR Evaluator   (this module)
                  |
                  v
             OCREngine      (src/ocr/base.py)
                  |
           +------+------+
           v             v
       Tesseract     PaddleOCR
         (now)        (future)
                  |
                  v
           OCR Prediction
                  |
           +------+------+
           v             v
          CER           WER    (src/ocr/metrics.py)
                  |
                  v
           Benchmark Results

Run with::

    python -m src.ocr.evaluation
    python -m src.ocr.evaluation --engine tesseract --dataset data/evaluation/dataset.json
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from src.ocr.base import OCREngine
from src.ocr.metrics import character_error_rate, tokenize_words, word_error_rate
from src.ocr.models import OCRError
from src.utils.config import OCR_CONFIG, OCRConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

#: Default locations, relative to the project root.
DEFAULT_DATASET_PATH = Path("data/evaluation/dataset.json")
DEFAULT_RESULTS_DIR = Path("data/evaluation/results")


class EvaluationError(Exception):
    """Raised when an evaluation dataset cannot be loaded or parsed."""


@dataclass(frozen=True)
class EvaluationSample:
    """One entry from the evaluation dataset manifest."""

    id: str
    image: str
    ground_truth: str
    language: str = "eng"
    category: str = "unspecified"
    notes: str = ""

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "EvaluationSample":
        missing = [k for k in ("id", "image", "ground_truth") if not raw.get(k)]
        if missing:
            raise EvaluationError(f"Dataset entry is missing required field(s): {missing} in {raw!r}")
        return cls(
            id=raw["id"],
            image=raw["image"],
            ground_truth=raw["ground_truth"],
            language=raw.get("language", "eng"),
            category=raw.get("category", "unspecified"),
            notes=raw.get("notes", ""),
        )


@dataclass(frozen=True)
class EvaluationResult:
    """
    Metrics for a single evaluated sample.

    ``success`` is False when OCR could not be run at all (missing
    image, missing ground truth, engine failure). In that case the
    metric fields stay ``None`` rather than being filled with
    placeholder numbers — a failed sample must never be silently
    averaged in as if it scored zero error.
    """

    dataset_id: str
    engine: str
    language: str
    category: str
    reference_text_length: int = 0
    recognized_text_length: int = 0
    reference_word_count: int = 0
    recognized_word_count: int = 0
    cer: Optional[float] = None
    wer: Optional[float] = None
    ocr_confidence: Optional[float] = None
    processing_time_ms: Optional[float] = None
    success: bool = False
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvaluationSummary:
    """Aggregate statistics across a set of :class:`EvaluationResult`."""

    engine: str
    total_samples: int
    successful_samples: int
    failed_samples: int
    average_cer: Optional[float] = None
    average_wer: Optional[float] = None
    median_cer: Optional[float] = None
    median_wer: Optional[float] = None
    average_confidence: Optional[float] = None
    average_processing_time_ms: Optional[float] = None
    fastest_sample: Optional[str] = None
    slowest_sample: Optional[str] = None
    by_category: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _load_image(path: Path) -> np.ndarray:
    """Load an evaluation image as a uint8 numpy array (the pipeline's standard type)."""
    # np.fromfile + imdecode rather than cv2.imread: imread silently
    # returns None for non-ASCII paths on some platforms.
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise EvaluationError(f"Could not decode image: {path}")
    return image


def evaluate_sample(
    sample: EvaluationSample,
    engine: OCREngine,
    base_dir: Path,
    config: OCRConfig = OCR_CONFIG,
) -> EvaluationResult:
    """
    Evaluate one sample: run OCR, compare against ground truth, return metrics.

    Contains no engine-specific logic — ``engine`` is used only through
    the :class:`~src.ocr.base.OCREngine` interface, so any future
    backend is scored by exactly this code path.

    Failures (missing files, OCR errors) are captured into the returned
    result with ``success=False`` rather than raised, so one bad sample
    cannot abort a whole benchmark run.
    """
    image_path = base_dir / sample.image
    truth_path = base_dir / sample.ground_truth

    def _failure(message: str) -> EvaluationResult:
        logger.error("Sample '%s' failed: %s", sample.id, message)
        return EvaluationResult(
            dataset_id=sample.id,
            engine=engine.name,
            language=sample.language,
            category=sample.category,
            success=False,
            error=message,
        )

    if not image_path.is_file():
        return _failure(f"Image not found: {image_path}")
    if not truth_path.is_file():
        return _failure(f"Ground truth not found: {truth_path}")

    try:
        reference = truth_path.read_text(encoding="utf-8")
    except OSError as exc:
        return _failure(f"Could not read ground truth: {exc}")

    try:
        image = _load_image(image_path)
    except EvaluationError as exc:
        return _failure(str(exc))

    try:
        ocr_result = engine.process(image, language=sample.language, config=config)
    except OCRError as exc:
        return _failure(f"OCR failed: {exc}")
    except Exception as exc:  # defensive: a new engine may raise its own type
        return _failure(f"Unexpected OCR failure: {exc}")

    hypothesis = ocr_result.text
    cer = character_error_rate(reference, hypothesis)
    wer = word_error_rate(reference, hypothesis)

    logger.info(
        "Sample '%s' [%s/%s]: CER=%.4f WER=%.4f confidence=%.1f time=%.1fms",
        sample.id,
        sample.category,
        sample.language,
        cer,
        wer,
        ocr_result.mean_confidence or 0.0,
        ocr_result.processing_time_ms or 0.0,
    )

    return EvaluationResult(
        dataset_id=sample.id,
        engine=ocr_result.engine,
        language=sample.language,
        category=sample.category,
        reference_text_length=len(reference),
        recognized_text_length=len(hypothesis),
        reference_word_count=len(tokenize_words(reference)),
        recognized_word_count=len(ocr_result.words),
        cer=cer,
        wer=wer,
        ocr_confidence=ocr_result.mean_confidence,
        processing_time_ms=ocr_result.processing_time_ms,
        success=True,
        error=None,
    )


def load_dataset(dataset_path: Path) -> List[EvaluationSample]:
    """Load and validate the dataset manifest."""
    if not dataset_path.is_file():
        raise EvaluationError(f"Dataset manifest not found: {dataset_path}")

    try:
        raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvaluationError(f"Dataset manifest is not valid JSON: {exc}") from exc

    entries = raw.get("samples", raw) if isinstance(raw, dict) else raw
    if not isinstance(entries, list):
        raise EvaluationError("Dataset manifest must contain a list of samples.")

    return [EvaluationSample.from_dict(entry) for entry in entries]


def summarize(results: List[EvaluationResult], engine_name: str) -> EvaluationSummary:
    """
    Build aggregate statistics from per-sample results.

    Only successful samples contribute to metric averages — failures are
    counted separately so a crashed sample never masquerades as a
    perfect score.
    """
    successful = [r for r in results if r.success]
    summary = EvaluationSummary(
        engine=engine_name,
        total_samples=len(results),
        successful_samples=len(successful),
        failed_samples=len(results) - len(successful),
    )

    if not successful:
        return summary

    cers = [r.cer for r in successful if r.cer is not None]
    wers = [r.wer for r in successful if r.wer is not None]
    confidences = [r.ocr_confidence for r in successful if r.ocr_confidence is not None]
    times = [(r.processing_time_ms, r.dataset_id) for r in successful if r.processing_time_ms is not None]

    if cers:
        summary.average_cer = statistics.fmean(cers)
        summary.median_cer = statistics.median(cers)
    if wers:
        summary.average_wer = statistics.fmean(wers)
        summary.median_wer = statistics.median(wers)
    if confidences:
        summary.average_confidence = statistics.fmean(confidences)
    if times:
        summary.average_processing_time_ms = statistics.fmean(t for t, _ in times)
        summary.fastest_sample = min(times)[1]
        summary.slowest_sample = max(times)[1]

    categories = {r.category for r in successful}
    for category in sorted(categories):
        subset = [r for r in successful if r.category == category]
        cat_cers = [r.cer for r in subset if r.cer is not None]
        cat_wers = [r.wer for r in subset if r.wer is not None]
        cat_conf = [r.ocr_confidence for r in subset if r.ocr_confidence is not None]
        cat_time = [r.processing_time_ms for r in subset if r.processing_time_ms is not None]
        summary.by_category[category] = {
            "samples": len(subset),
            "average_cer": statistics.fmean(cat_cers) if cat_cers else None,
            "average_wer": statistics.fmean(cat_wers) if cat_wers else None,
            "average_confidence": statistics.fmean(cat_conf) if cat_conf else None,
            "average_processing_time_ms": statistics.fmean(cat_time) if cat_time else None,
        }

    return summary


def save_results(
    results: List[EvaluationResult],
    summary: EvaluationSummary,
    results_dir: Path,
) -> Dict[str, Path]:
    """Write ``results.json`` and ``results.csv`` to ``results_dir``."""
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / "results.json"
    csv_path = results_dir / "results.csv"

    json_path.write_text(
        json.dumps(
            {"summary": summary.to_dict(), "results": [r.to_dict() for r in results]},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    fieldnames = list(EvaluationResult.__dataclass_fields__.keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result.to_dict())

    logger.info("Saved evaluation results to %s and %s", json_path, csv_path)
    return {"json": json_path, "csv": csv_path}


def evaluate_dataset(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    engine: Optional[OCREngine] = None,
    engine_name: Optional[str] = None,
    results_dir: Optional[Path] = DEFAULT_RESULTS_DIR,
    config: OCRConfig = OCR_CONFIG,
) -> tuple[List[EvaluationResult], EvaluationSummary]:
    """
    Run a full benchmark over every sample in the dataset manifest.

    Args:
        dataset_path: Path to the dataset JSON manifest.
        engine: An explicit engine instance (used by tests to inject a
            fake engine). When omitted, one is built by name via the
            factory — so ``evaluate_dataset(engine_name="paddleocr")``
            will work unchanged once that engine exists.
        engine_name: Engine to build when ``engine`` is not supplied.
        results_dir: Where to write results.json/results.csv. Pass
            ``None`` to skip writing to disk.
        config: OCR configuration.
    """
    if engine is None:
        from src.ocr.factory import get_ocr_engine

        engine = get_ocr_engine(engine_name, config)

    samples = load_dataset(dataset_path)
    base_dir = dataset_path.parent
    logger.info("Evaluating %d sample(s) with engine '%s'", len(samples), engine.name)

    results = [evaluate_sample(s, engine, base_dir, config) for s in samples]
    summary = summarize(results, engine.name)

    if results_dir is not None:
        save_results(results, summary, results_dir)

    return results, summary


def _format_summary(summary: EvaluationSummary) -> str:
    """Render a summary as readable terminal text."""

    def fmt(value: Optional[float], suffix: str = "", places: int = 4) -> str:
        return "n/a" if value is None else f"{value:.{places}f}{suffix}"

    lines = [
        "",
        "=" * 60,
        "OCR EVALUATION SUMMARY",
        "=" * 60,
        f"Engine:                {summary.engine}",
        f"Samples:               {summary.total_samples}",
        f"Successful:            {summary.successful_samples}",
        f"Failed:                {summary.failed_samples}",
        f"Average CER:           {fmt(summary.average_cer)}",
        f"Median CER:            {fmt(summary.median_cer)}",
        f"Average WER:           {fmt(summary.average_wer)}",
        f"Median WER:            {fmt(summary.median_wer)}",
        f"Average confidence:    {fmt(summary.average_confidence, '%', 2)}",
        f"Average time:          {fmt(summary.average_processing_time_ms, 'ms', 1)}",
        f"Fastest sample:        {summary.fastest_sample or 'n/a'}",
        f"Slowest sample:        {summary.slowest_sample or 'n/a'}",
    ]

    if summary.by_category:
        lines.append("-" * 60)
        lines.append("BY CATEGORY")
        for category, stats in summary.by_category.items():
            lines.append(f"  Category: {category}")
            lines.append(f"    Samples:            {stats['samples']}")
            lines.append(f"    Average CER:        {fmt(stats['average_cer'])}")
            lines.append(f"    Average WER:        {fmt(stats['average_wer'])}")
            lines.append(f"    Average confidence: {fmt(stats['average_confidence'], '%', 2)}")
            lines.append(f"    Average time:       {fmt(stats['average_processing_time_ms'], 'ms', 1)}")

    lines.append("=" * 60)
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point: ``python -m src.ocr.evaluation``."""
    parser = argparse.ArgumentParser(
        prog="python -m src.ocr.evaluation",
        description="Run the OCR evaluation benchmark (CER/WER) against a dataset.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f"Path to the dataset manifest (default: {DEFAULT_DATASET_PATH}).",
    )
    parser.add_argument(
        "--engine",
        default=None,
        help="OCR engine name (default: OCRConfig.engine / OCR_ENGINE env var).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Directory for results.json/results.csv (default: {DEFAULT_RESULTS_DIR}).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Print the summary without writing result files.",
    )
    args = parser.parse_args(argv)

    try:
        _, summary = evaluate_dataset(
            dataset_path=args.dataset,
            engine_name=args.engine,
            results_dir=None if args.no_save else args.results_dir,
        )
    except (EvaluationError, OCRError) as exc:
        print(f"Evaluation failed: {exc}")
        return 1

    print(_format_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
