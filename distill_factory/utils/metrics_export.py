"""Teacher-quality metrics export utilities."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import orjson
except ModuleNotFoundError:  # pragma: no cover
    orjson = None
    import json


METRICS_SCHEMA_VERSION = "1.0"
METRICS_EXPORT_FILENAME = "teacher_quality_metrics.json"


def _dumps(data: dict[str, Any]) -> bytes:
    if orjson is not None:
        return orjson.dumps(data)
    return json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")


def _loads(blob: bytes) -> dict[str, Any]:
    if orjson is not None:
        return orjson.loads(blob)
    return json.loads(blob.decode("utf-8"))


def _histogram(values: list[float], bins: int) -> list[dict[str, Any]]:
    if not values:
        return []

    n_bins = max(1, int(bins))
    low = min(values)
    high = max(values)
    if high == low:
        return [{"bin_start": low, "bin_end": high, "count": len(values)}]

    width = (high - low) / n_bins
    counts = [0 for _ in range(n_bins)]
    for v in values:
        idx = int((v - low) / width)
        if idx >= n_bins:
            idx = n_bins - 1
        counts[idx] += 1

    out: list[dict[str, Any]] = []
    for i, count in enumerate(counts):
        b_start = low + (i * width)
        b_end = high if i == n_bins - 1 else low + ((i + 1) * width)
        out.append({"bin_start": b_start, "bin_end": b_end, "count": count})
    return out



def _selection_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    if total == 0:
        return {
            "selected_window_record_count": 0,
            "average_selected_window_length": None,
            "average_selected_position_count": None,
            "position_masks_present": False,
            "selection_filtered_record_proportion": 0.0,
        }

    selected_window_lengths: list[int] = []
    selected_position_counts: list[int] = []
    selected_window_record_count = 0
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
    avg_selected_positions = (
        float(sum(selected_position_counts) / len(selected_position_counts)) if selected_position_counts else None
    )
    proportion = float(selection_filtered_record_count / total) if total > 0 else 0.0

    return {
        "selected_window_record_count": int(selected_window_record_count),
        "average_selected_window_length": avg_window_len,
        "average_selected_position_count": avg_selected_positions,
        "position_masks_present": bool(position_masks_present),
        "selection_filtered_record_proportion": proportion,
    }

def build_teacher_quality_summary(records: list[dict[str, Any]], bins: int = 10) -> dict[str, Any]:
    entropies = [float(r["entropy"]) for r in records if r.get("entropy") is not None]

    stage_counts: dict[str, int] = {}
    teacher_counts: dict[str, int] = {}
    stage_entropy: dict[str, list[float]] = {}
    teacher_entropy: dict[str, list[float]] = {}

    for r in records:
        stage = str(r.get("stage_name", "missing"))
        teacher = str(r.get("teacher_name", "missing"))

        stage_counts[stage] = stage_counts.get(stage, 0) + 1
        teacher_counts[teacher] = teacher_counts.get(teacher, 0) + 1

        ent = r.get("entropy")
        if ent is not None:
            val = float(ent)
            stage_entropy.setdefault(stage, []).append(val)
            teacher_entropy.setdefault(teacher, []).append(val)

    avg_entropy_per_stage = {
        k: (sum(v) / len(v)) for k, v in stage_entropy.items() if v
    }
    avg_entropy_per_teacher = {
        k: (sum(v) / len(v)) for k, v in teacher_entropy.items() if v
    }

    return {
        "schema_version": METRICS_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "record_count": len(records),
        "entropy_histogram": {
            "binning": "deterministic_equal_width",
            "bin_count": max(1, int(bins)),
            "bins": _histogram(entropies, bins=bins),
        },
        "average_entropy_per_stage": avg_entropy_per_stage,
        "average_entropy_per_teacher": avg_entropy_per_teacher,
        "record_counts_per_stage": stage_counts,
        "record_counts_per_teacher": teacher_counts,
        "selection_summary": _selection_summary(records),
    }


def export_teacher_quality_metrics(
    records: list[dict[str, Any]],
    output_dir: str | Path,
    bins: int = 10,
    filename: str = METRICS_EXPORT_FILENAME,
) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = build_teacher_quality_summary(records, bins=bins)
    out_path = out_dir / filename
    out_path.write_bytes(_dumps(summary))
    return out_path


def load_teacher_quality_metrics(path: str | Path) -> dict[str, Any] | None:
    p = Path(path)
    candidate = p if p.is_file() else p / METRICS_EXPORT_FILENAME
    if not candidate.exists():
        return None
    try:
        return _loads(candidate.read_bytes())
    except Exception:
        return None
