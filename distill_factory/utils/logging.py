"""Logging helpers."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
from typing import Any

from distill_factory.utils.metrics_export import export_teacher_quality_metrics


FAILURE_LOG_FILENAME = "record_failures.jsonl"


def get_logger(name: str = "distill_factory") -> logging.Logger:
    """Return a package logger with a basic configuration."""
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(name)


def append_record_failure(
    output_dir: str | Path,
    *,
    stage_name: str,
    teacher_name: str,
    error_message: str,
    doc_id: str | None = None,
    chunk_index: int | None = None,
) -> Path:
    """Append a single record-level failure entry to the dataset failure log."""
    path = Path(output_dir) / FAILURE_LOG_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stage_name": stage_name,
        "teacher_name": teacher_name,
        "doc_id": doc_id,
        "chunk_index": chunk_index,
        "error_message": error_message,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return path


def _topk_width(top_k_ids: Any) -> float:
    if top_k_ids is None:
        return 0.0
    if isinstance(top_k_ids, list):
        if not top_k_ids:
            return 0.0
        if isinstance(top_k_ids[0], list):
            widths = [len(row) for row in top_k_ids if isinstance(row, list)]
            return float(sum(widths) / len(widths)) if widths else 0.0
        return float(len(top_k_ids))
    return 0.0


def _token_length(top_k_ids: Any) -> int:
    if top_k_ids is None:
        return 0
    if isinstance(top_k_ids, list):
        if not top_k_ids:
            return 0
        if isinstance(top_k_ids[0], list):
            return len(top_k_ids)
        return 1
    return 0


def _teacher_input_byte_length(record: dict[str, Any]) -> int:
    explicit = record.get("teacher_input_byte_length")
    if isinstance(explicit, (int, float)):
        return int(explicit)

    meta = record.get("extra_metadata")
    if isinstance(meta, dict):
        ctx = meta.get("stage_b_context")
        if isinstance(ctx, dict):
            text = ctx.get("long_context_text")
            if isinstance(text, str):
                return len(text.encode("utf-8", errors="replace"))
    raw = record.get("raw_bytes", b"")
    if isinstance(raw, bytes):
        return len(raw)
    if isinstance(raw, str):
        return len(raw.encode("utf-8", errors="replace"))
    return len(str(raw).encode("utf-8", errors="replace"))


def summarize_teacher_outputs(records: list[dict[str, Any]], stage_name: str) -> dict[str, Any]:
    """Compute simple teacher-output sanity metrics for a stage."""
    if not records:
        return {
            "stage_name": stage_name,
            "record_count": 0,
            "teacher_counts": {},
            "avg_entropy": None,
            "min_entropy": None,
            "max_entropy": None,
            "avg_chunk_length": 0.0,
            "avg_topk_width": 0.0,
            "avg_teacher_input_bytes": None,
            "avg_teacher_input_tokens": None,
            "min_teacher_input_tokens": None,
            "max_teacher_input_tokens": None,
            "avg_bytes_per_token": None,
        }

    entropies = [float(r["entropy"]) for r in records if r.get("entropy") is not None]
    chunk_lengths = [int(r["byte_end"]) - int(r["byte_start"]) for r in records if "byte_end" in r and "byte_start" in r]
    topk_widths = [_topk_width(r.get("top_k_ids")) for r in records if r.get("top_k_ids") is not None]

    teacher_counts: dict[str, int] = {}
    for r in records:
        teacher = str(r.get("teacher_name", "missing"))
        teacher_counts[teacher] = teacher_counts.get(teacher, 0) + 1

    log_tokens = os.environ.get("DISTILL_FACTORY_LOG_TOKEN_LENGTHS", "0") == "1"
    log_bytes = os.environ.get("DISTILL_FACTORY_LOG_BYTE_LENGTHS", "0") == "1"

    byte_lengths: list[int] = []
    token_lengths: list[int] = []
    if log_bytes:
        byte_lengths = [_teacher_input_byte_length(r) for r in records]
    if log_tokens:
        explicit_token_lengths = [
            int(r["teacher_input_token_length"]) for r in records if isinstance(r.get("teacher_input_token_length"), (int, float))
        ]
        if explicit_token_lengths:
            token_lengths = explicit_token_lengths
        else:
            token_lengths = [_token_length(r.get("top_k_ids")) for r in records if r.get("top_k_ids") is not None]

    avg_bytes = (sum(byte_lengths) / len(byte_lengths)) if byte_lengths else None
    avg_tokens = (sum(token_lengths) / len(token_lengths)) if token_lengths else None
    min_tokens = min(token_lengths) if token_lengths else None
    max_tokens = max(token_lengths) if token_lengths else None

    ratio = None
    if avg_bytes is not None and avg_tokens is not None and avg_tokens > 0:
        ratio = avg_bytes / avg_tokens

    return {
        "stage_name": stage_name,
        "record_count": len(records),
        "teacher_counts": teacher_counts,
        "avg_entropy": (sum(entropies) / len(entropies)) if entropies else None,
        "min_entropy": min(entropies) if entropies else None,
        "max_entropy": max(entropies) if entropies else None,
        "avg_chunk_length": (sum(chunk_lengths) / len(chunk_lengths)) if chunk_lengths else 0.0,
        "avg_topk_width": (sum(topk_widths) / len(topk_widths)) if topk_widths else 0.0,
        "avg_teacher_input_bytes": avg_bytes,
        "avg_teacher_input_tokens": avg_tokens,
        "min_teacher_input_tokens": min_tokens,
        "max_teacher_input_tokens": max_tokens,
        "avg_bytes_per_token": ratio,
    }


def _try_export_metrics(records: list[dict[str, Any]], logger: logging.Logger) -> str | None:
    output_dir = os.environ.get("DISTILL_FACTORY_OUTPUT_DIR")
    if not output_dir:
        return None
    try:
        metrics_path = export_teacher_quality_metrics(records=records, output_dir=Path(output_dir))
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to export teacher-quality metrics: %s", exc)
        return None
    return str(metrics_path)


def log_stage_metrics(records: list[dict[str, Any]], stage_name: str, logger_name: str = "distill_factory") -> dict[str, Any]:
    """Log and return stage-level sanity metrics."""
    logger = get_logger(logger_name)
    summary = summarize_teacher_outputs(records, stage_name=stage_name)
    logger.info(
        "%s metrics | records=%d avg_entropy=%s min_entropy=%s max_entropy=%s avg_chunk_len=%.2f avg_topk_width=%.2f avg_teacher_input_bytes=%s avg_teacher_input_tokens=%s min_teacher_input_tokens=%s max_teacher_input_tokens=%s avg_bytes_per_token=%s teacher_counts=%s",
        stage_name,
        summary["record_count"],
        f"{summary['avg_entropy']:.4f}" if summary["avg_entropy"] is not None else "n/a",
        f"{summary['min_entropy']:.4f}" if summary["min_entropy"] is not None else "n/a",
        f"{summary['max_entropy']:.4f}" if summary["max_entropy"] is not None else "n/a",
        summary["avg_chunk_length"],
        summary["avg_topk_width"],
        f"{summary['avg_teacher_input_bytes']:.2f}" if summary["avg_teacher_input_bytes"] is not None else "n/a",
        f"{summary['avg_teacher_input_tokens']:.2f}" if summary["avg_teacher_input_tokens"] is not None else "n/a",
        summary["min_teacher_input_tokens"] if summary["min_teacher_input_tokens"] is not None else "n/a",
        summary["max_teacher_input_tokens"] if summary["max_teacher_input_tokens"] is not None else "n/a",
        f"{summary['avg_bytes_per_token']:.4f}" if summary["avg_bytes_per_token"] is not None else "n/a",
        summary["teacher_counts"],
    )

    metrics_path = _try_export_metrics(records, logger)
    if metrics_path is not None:
        logger.info("teacher quality metrics exported: %s", metrics_path)
        summary["metrics_export_path"] = metrics_path

    return summary


def format_timing_report(summary: dict[str, Any]) -> list[str]:
    """Return human-readable timing lines from pipeline summary timing payload."""
    timing = summary.get("timing") if isinstance(summary, dict) else None
    if not isinstance(timing, dict):
        return ["timing: unavailable"]

    lines = [
        f"ingestion_seconds={float(timing.get('ingestion_seconds', 0.0)):.4f}",
        f"chunking_seconds={float(timing.get('chunking_seconds', 0.0)):.4f}",
        f"splitting_seconds={float(timing.get('splitting_seconds', 0.0)):.4f}",
        f"stage_a_seconds={float(timing.get('stage_a_seconds', 0.0)):.4f}",
        f"stage_b_seconds={float(timing.get('stage_b_seconds', 0.0)):.4f}",
        f"stage_c_seconds={float(timing.get('stage_c_seconds', 0.0)):.4f}",
        f"teacher_inference_seconds={float(timing.get('teacher_inference_seconds', 0.0)):.4f}",
        f"writing_seconds={float(timing.get('writing_seconds', 0.0)):.4f}",
        f"total_runtime_seconds={float(timing.get('total_runtime_seconds', 0.0)):.4f}",
    ]
    return lines
