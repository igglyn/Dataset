#!/usr/bin/env python3
"""Inspect generated distillation datasets from JSONL files."""

from __future__ import annotations

import argparse
import base64
import json
import sys
from collections import Counter
from pathlib import Path
from pprint import pformat
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from distill_factory.storage.reader import (
    get_shard_stats,
    load_dataset_manifest,
    read_jsonl_records,
    sample_jsonl_records,
)
from distill_factory.utils.metrics_export import (
    METRICS_EXPORT_FILENAME,
    export_teacher_quality_metrics,
    load_teacher_quality_metrics,
)


CANONICAL_FIELDS = [
    "schema_version",
    "doc_id",
    "chunk_index",
    "byte_start",
    "byte_end",
    "raw_bytes_b64",
    "split",
    "teacher_name",
    "stage_name",
    "mode",
    "top_k_ids",
    "top_k_logprobs",
    "entropy",
    "extra_metadata",
]




def _raw_text_preview(record: dict[str, Any], max_chars: int = 240) -> str:
    raw = record.get("raw_bytes")
    if isinstance(raw, bytes):
        text = raw.decode("utf-8", errors="replace")
    elif isinstance(raw, str):
        text = raw
    else:
        raw_b64 = record.get("raw_bytes_b64")
        if isinstance(raw_b64, str) and raw_b64:
            try:
                text = base64.b64decode(raw_b64.encode("ascii")).decode("utf-8", errors="replace")
            except Exception:
                text = ""
        else:
            text = ""
    text = text.replace("\n", "\\n")
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text


def _topk_preview(record: dict[str, Any]) -> dict[str, Any] | None:
    ids = record.get("top_k_ids")
    lps = record.get("top_k_logprobs")
    if ids is None and lps is None:
        return None

    width = _topk_width(ids)
    first_ids: list[Any] = []
    first_lps: list[Any] = []
    if isinstance(ids, list):
        if ids and isinstance(ids[0], list):
            first_ids = ids[0][:5]
        else:
            first_ids = ids[:5]
    if isinstance(lps, list):
        if lps and isinstance(lps[0], list):
            first_lps = lps[0][:5]
        else:
            first_lps = lps[:5]

    return {
        "avg_width": float(width),
        "first_position_ids": first_ids,
        "first_position_logprobs": first_lps,
    }


def _structured_preview(record: dict[str, Any], max_chars: int = 240) -> dict[str, Any] | None:
    structured = record.get("structured_output")
    if not isinstance(structured, dict):
        meta = record.get("extra_metadata")
        if isinstance(meta, dict) and isinstance(meta.get("structured_output"), dict):
            structured = meta.get("structured_output")
    if not isinstance(structured, dict):
        return None

    completion = str(structured.get("completion_text", ""))
    if len(completion) > max_chars:
        completion = completion[:max_chars] + "..."
    return {
        "task_type": structured.get("task_type"),
        "prompt_preview": str(structured.get("prompt_text", ""))[:max_chars],
        "completion_preview": completion,
    }


def _sample_export_payload(record: dict[str, Any], idx: int) -> dict[str, Any]:
    return {
        "sample_index": idx,
        "metadata": {
            "doc_id": record.get("doc_id"),
            "chunk_index": record.get("chunk_index"),
            "byte_start": record.get("byte_start"),
            "byte_end": record.get("byte_end"),
            "split": record.get("split"),
            "stage_name": record.get("stage_name"),
            "teacher_name": record.get("teacher_name"),
            "mode": record.get("mode"),
        },
        "raw_text_preview": _raw_text_preview(record),
        "top_k_summary": _topk_preview(record),
        "structured_output_preview": _structured_preview(record),
        "entropy_preview": record.get("entropy"),
        "extra_metadata": record.get("extra_metadata"),
    }


def _write_sample_export(
    *,
    source_path: str,
    output_path: str,
    sample_count: int,
    sample_stage: str | None,
    sample_teacher: str | None,
    output_format: str,
) -> int:
    selected = sample_jsonl_records(
        source_path,
        n=sample_count,
        stage_name=sample_stage,
        teacher_name=sample_teacher,
    )
    payload = [_sample_export_payload(rec, idx=i) for i, rec in enumerate(selected)]

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if output_format == "json":
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        lines: list[str] = []
        lines.append(f"sample_count={len(payload)}")
        lines.append(f"source_path={source_path}")
        lines.append(f"filter_stage={sample_stage}")
        lines.append(f"filter_teacher={sample_teacher}")
        lines.append("")
        for item in payload:
            lines.append(f"=== Sample {item['sample_index']} ===")
            lines.append("metadata:")
            for k, v in item["metadata"].items():
                lines.append(f"  {k}: {v}")
            lines.append(f"entropy_preview: {item['entropy_preview']}")
            lines.append(f"raw_text_preview: {item['raw_text_preview']}")
            lines.append(f"top_k_summary: {pformat(item['top_k_summary'], sort_dicts=True)}")
            lines.append(f"structured_output_preview: {pformat(item['structured_output_preview'], sort_dicts=True)}")
            lines.append("")
        out.write_text("\n".join(lines), encoding="utf-8")

    return len(payload)


def _avg_chunk_len(records: list[dict[str, Any]]) -> float:
    if not records:
        return 0.0
    return sum(int(r["byte_end"]) - int(r["byte_start"]) for r in records) / len(records)


def _topk_width(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, dict) and value.get("shape"):
        shape = value.get("shape")
        if isinstance(shape, list) and len(shape) >= 1:
            if len(shape) == 1:
                return float(shape[0])
            return float(shape[-1])
    if isinstance(value, list):
        if not value:
            return 0.0
        if isinstance(value[0], list):
            widths = [len(row) for row in value if isinstance(row, list)]
            return float(sum(widths) / len(widths)) if widths else 0.0
        return float(len(value))
    return 0.0


def _entropy_stats(records: list[dict[str, Any]]) -> tuple[float | None, float | None, float | None]:
    vals = [float(r["entropy"]) for r in records if r.get("entropy") is not None]
    if not vals:
        return None, None, None
    return sum(vals) / len(vals), min(vals), max(vals)


def _tokenization_cost_stats(records: list[dict[str, Any]]) -> tuple[float | None, float | None, int | None, int | None]:
    token_lengths = [
        int(r["teacher_input_token_length"])
        for r in records
        if isinstance(r.get("teacher_input_token_length"), (int, float))
    ]
    if not token_lengths:
        return None, None, None, None

    byte_lengths = [
        int(r.get("teacher_input_byte_length", 0))
        for r in records
        if isinstance(r.get("teacher_input_byte_length"), (int, float))
    ]
    avg_tokens = sum(token_lengths) / len(token_lengths)
    avg_bytes = (sum(byte_lengths) / len(byte_lengths)) if byte_lengths else None
    ratio = (avg_bytes / avg_tokens) if (avg_bytes is not None and avg_tokens > 0) else None
    return avg_tokens, ratio, min(token_lengths), max(token_lengths)


def _avg_topk_width(records: list[dict[str, Any]]) -> float:
    widths = [_topk_width(r.get("top_k_ids")) for r in records if r.get("top_k_ids") is not None]
    return (sum(widths) / len(widths)) if widths else 0.0


def _has_topk(records: list[dict[str, Any]]) -> bool:
    return any(r.get("top_k_ids") is not None or r.get("top_k_logprobs") is not None for r in records)


def _has_structured(records: list[dict[str, Any]]) -> bool:
    for r in records:
        if r.get("structured_output") is not None:
            return True
        meta = r.get("extra_metadata")
        if isinstance(meta, dict) and meta.get("structured_output") is not None:
            return True
    return False


def _print_counter(title: str, counter: Counter[str]) -> None:
    print(title)
    if not counter:
        print("  (none)")
        return
    for key, value in counter.most_common():
        print(f"  {key}: {value}")


def _print_teacher_sanity_if_present(records: list[dict[str, Any]]) -> None:
    sanity = None
    for r in records:
        meta = r.get("extra_metadata")
        if isinstance(meta, dict) and isinstance(meta.get("teacher_sanity"), dict):
            sanity = meta["teacher_sanity"]
            break
    if sanity is None:
        return

    print("\nEmbedded teacher sanity metrics (from pipeline run)")
    print(f"  stage_name: {sanity.get('stage_name')}")
    print(f"  record_count: {sanity.get('record_count')}")
    print(f"  avg_entropy: {sanity.get('avg_entropy')}")
    print(f"  min_entropy: {sanity.get('min_entropy')}")
    print(f"  max_entropy: {sanity.get('max_entropy')}")
    print(f"  avg_chunk_length: {sanity.get('avg_chunk_length')}")
    print(f"  avg_topk_width: {sanity.get('avg_topk_width')}")
    print(f"  teacher_counts: {sanity.get('teacher_counts')}")


def _print_manifest(path: str) -> None:
    manifest = load_dataset_manifest(path)
    print("\nDataset manifest")
    if manifest is None:
        print("  (not found)")
        return

    keys = [
        "schema_version",
        "creation_timestamp",
        "updated_timestamp",
        "config_path",
        "enabled_stages",
        "teacher_names",
        "shard_count",
        "total_record_count",
        "format",
        "compression",
        "merged_from",
    ]
    for key in keys:
        print(f"  {key}: {manifest.get(key)}")

    settings = manifest.get("format_settings")
    if settings is not None:
        print(f"  format_settings: {settings}")


def _print_shard_stats(path: str) -> None:
    stats = get_shard_stats(path)
    print("\nShard stats")
    print(f"  Shard count: {len(stats)}")
    for shard in stats:
        print(f"  - {shard['path']}: {shard['record_count']}")



def _selection_inspection(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    if total == 0:
        return {
            "selected_window_record_count": 0,
            "average_selected_window_length": None,
            "average_selected_position_count": None,
            "position_masks_present": False,
            "selection_filtered_record_proportion": 0.0,
        }

    selected_window_record_count = 0
    selected_window_lengths: list[int] = []
    selected_position_counts: list[int] = []
    position_masks_present = False
    selection_filtered_record_count = 0

    for r in records:
        meta = r.get("extra_metadata")
        if not isinstance(meta, dict):
            continue

        has_selection_marker = (
            "selected_window_start" in meta
            or "selected_position_mask" in meta
            or "selection_policy" in meta
            or "selected_positions" in meta
        )
        if has_selection_marker:
            selection_filtered_record_count += 1

        if "selected_position_mask" in meta:
            position_masks_present = True

        if "selected_window_start" in meta and "selected_window_end" in meta:
            try:
                start = int(meta["selected_window_start"])
                end = int(meta["selected_window_end"])
                selected_window_record_count += 1
                selected_window_lengths.append(max(0, end - start + 1))
            except Exception:
                pass

        if "selected_position_count" in meta:
            try:
                selected_position_counts.append(int(meta["selected_position_count"]))
            except Exception:
                pass

    avg_window_len = (
        float(sum(selected_window_lengths) / len(selected_window_lengths)) if selected_window_lengths else None
    )
    avg_selected_pos = (
        float(sum(selected_position_counts) / len(selected_position_counts)) if selected_position_counts else None
    )

    return {
        "selected_window_record_count": int(selected_window_record_count),
        "average_selected_window_length": avg_window_len,
        "average_selected_position_count": avg_selected_pos,
        "position_masks_present": bool(position_masks_present),
        "selection_filtered_record_proportion": float(selection_filtered_record_count / total),
    }

def _dataset_dir(path: str) -> Path:
    p = Path(path)
    return p if p.is_dir() else p.parent


def _load_or_export_metrics(path: str, records: list[dict[str, Any]], bins: int) -> tuple[dict[str, Any] | None, Path]:
    directory = _dataset_dir(path)
    metrics_path = directory / METRICS_EXPORT_FILENAME
    metrics = load_teacher_quality_metrics(metrics_path)
    if metrics is None and records:
        export_teacher_quality_metrics(records, directory, bins=bins)
        metrics = load_teacher_quality_metrics(metrics_path)
    return metrics, metrics_path


def _print_metrics_export(metrics: dict[str, Any] | None, metrics_path: Path, show_histogram: bool) -> None:
    print("\nTeacher-quality metrics export")
    print(f"  path: {metrics_path}")
    print(f"  present: {metrics is not None}")
    if metrics is None:
        return

    print(f"  schema_version: {metrics.get('schema_version')}")
    print(f"  record_count: {metrics.get('record_count')}")
    print(f"  average_entropy_per_stage: {metrics.get('average_entropy_per_stage')}")
    print(f"  average_entropy_per_teacher: {metrics.get('average_entropy_per_teacher')}")
    print(f"  record_counts_per_stage: {metrics.get('record_counts_per_stage')}")
    print(f"  record_counts_per_teacher: {metrics.get('record_counts_per_teacher')}")
    print(f"  selection_summary: {metrics.get('selection_summary')}")

    if not show_histogram:
        return

    hist = metrics.get("entropy_histogram", {}) if isinstance(metrics, dict) else {}
    bins = hist.get("bins", []) if isinstance(hist, dict) else []
    print("  entropy_histogram_bins:")
    if not bins:
        print("    (none)")
        return
    for b in bins:
        print(
            "    "
            f"[{float(b.get('bin_start', 0.0)):.4f}, {float(b.get('bin_end', 0.0)):.4f}]"
            f": {int(b.get('count', 0))}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a distillation dataset JSONL file or shard directory.")
    parser.add_argument("--path", required=True, help="Path to dataset JSONL file or directory.")
    parser.add_argument("--stage", default=None, help="Optional stage filter before computing summary stats (for example: stage_a).")
    parser.add_argument("--histogram-bins", type=int, default=10, help="Number of deterministic entropy histogram bins.")
    parser.add_argument("--show-histogram", action="store_true", help="Print histogram bins from metrics export.")
    parser.add_argument("--sample-count", type=int, default=3, help="Number of sample records to export when --export-samples is set.")
    parser.add_argument("--sample-stage", default=None, help="Optional stage filter for sample export (for example: stage_a).")
    parser.add_argument("--sample-teacher", default=None, help="Optional teacher filter for sample export.")
    parser.add_argument("--export-samples", default=None, help="Optional output file path for sample export.")
    parser.add_argument("--sample-format", choices=["text", "json"], default="text", help="Output format for sample export file.")
    args = parser.parse_args()

    records = read_jsonl_records(args.path)
    if args.stage is not None:
        records = [r for r in records if str(r.get("stage_name", "")) == args.stage]

    print("=== Dataset Inspection ===")
    print(f"Path: {args.path}")
    print(f"Record count: {len(records)}")
    print(f"Stage filter: {args.stage}")
    print("Compare with another run: python scripts/compare_datasets.py --left <path_a> --right <path_b>")

    _print_shard_stats(args.path)
    _print_manifest(args.path)

    metrics, metrics_path = _load_or_export_metrics(args.path, records, bins=args.histogram_bins)
    _print_metrics_export(metrics, metrics_path, show_histogram=args.show_histogram)

    if args.export_samples is not None:
        exported = _write_sample_export(
            source_path=args.path,
            output_path=args.export_samples,
            sample_count=max(0, int(args.sample_count)),
            sample_stage=args.sample_stage,
            sample_teacher=args.sample_teacher,
            output_format=args.sample_format,
        )
        print(f"\nSample export written: {args.export_samples} (records: {exported})")

    if not records:
        print("Dataset is empty.")
        return

    schema_versions = sorted({str(r.get("schema_version", "missing")) for r in records})
    observed_keys = sorted({k for r in records for k in r.keys()})

    print("\nSchema summary")
    print(f"  Versions: {', '.join(schema_versions)}")
    print(f"  Canonical fields: {', '.join(CANONICAL_FIELDS)}")
    print(f"  Observed fields: {', '.join(observed_keys)}")

    stage_counts = Counter(str(r.get("stage_name", "missing")) for r in records)
    teacher_counts = Counter(str(r.get("teacher_name", "missing")) for r in records)

    print()
    _print_counter("Count by stage", stage_counts)
    print()
    _print_counter("Count by teacher", teacher_counts)

    avg_entropy, min_entropy, max_entropy = _entropy_stats(records)

    print("\nTeacher sanity metrics")
    print(f"  Average entropy: {'n/a' if avg_entropy is None else f'{avg_entropy:.4f}'}")
    print(f"  Min entropy: {'n/a' if min_entropy is None else f'{min_entropy:.4f}'}")
    print(f"  Max entropy: {'n/a' if max_entropy is None else f'{max_entropy:.4f}'}")
    print(f"  Average chunk length (bytes): {_avg_chunk_len(records):.2f}")
    print(f"  Average top-k width populated: {_avg_topk_width(records):.2f}")

    avg_tokens, bytes_per_token, min_tokens, max_tokens = _tokenization_cost_stats(records)
    print("\nTokenization-cost diagnostics")
    print(f"  Average teacher input tokens: {'n/a' if avg_tokens is None else f'{avg_tokens:.2f}'}")
    print(f"  Min teacher input tokens: {'n/a' if min_tokens is None else min_tokens}")
    print(f"  Max teacher input tokens: {'n/a' if max_tokens is None else max_tokens}")
    print(f"  Average bytes/token ratio: {'n/a' if bytes_per_token is None else f'{bytes_per_token:.4f}'}")

    _print_teacher_sanity_if_present(records)

    print("\nField presence")
    print(f"  top_k fields present: {_has_topk(records)}")
    print(f"  structured fields present: {_has_structured(records)}")

    selection = _selection_inspection(records)
    print("\nSelection-aware export inspection")
    print(f"  selected-window record count: {selection['selected_window_record_count']}")
    avg_window = selection['average_selected_window_length']
    print(f"  average selected window length: {'n/a' if avg_window is None else f'{avg_window:.2f}'}")
    avg_positions = selection['average_selected_position_count']
    print(f"  average selected position count: {'n/a' if avg_positions is None else f'{avg_positions:.2f}'}")
    print(f"  position masks present: {selection['position_masks_present']}")
    print(f"  selection-filtered record proportion: {selection['selection_filtered_record_proportion']:.4f}")

    print("\nSample record")
    print(pformat(records[0], sort_dicts=True, width=100))


if __name__ == "__main__":
    main()
