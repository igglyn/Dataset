"""Storage reading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from distill_factory.data.formats import DistilledSample, from_record

try:
    import orjson
except ModuleNotFoundError:  # pragma: no cover
    orjson = None
    import json


_MANIFEST_NAME = "dataset_manifest.json"
_RESUME_STATE_NAME = "resume_state.json"


def _loads(line: bytes) -> dict[str, Any]:
    if orjson is not None:
        return orjson.loads(line)
    return json.loads(line.decode("utf-8"))


def _resolve_jsonl_inputs(path: str | Path) -> list[Path]:
    p = Path(path)

    if p.is_dir():
        return sorted(p.glob("*.jsonl"))

    if p.exists():
        return [p]

    # Compatibility: if base file is missing, look for deterministic shard pattern.
    shard_candidates = sorted(p.parent.glob(f"{p.stem}-*.jsonl"))
    if shard_candidates:
        return shard_candidates

    return []


def _resolve_dataset_dir(path: str | Path) -> Path:
    p = Path(path)
    if p.is_dir():
        return p
    if p.exists():
        return p.parent

    if sorted(p.parent.glob(f"{p.stem}-*.jsonl")):
        return p.parent
    return p.parent


def load_dataset_manifest(path: str | Path) -> dict[str, Any] | None:
    """Load dataset manifest metadata if present."""
    manifest_path = _resolve_dataset_dir(path) / _MANIFEST_NAME
    if not manifest_path.exists():
        return None
    try:
        raw = manifest_path.read_bytes()
        return _loads(raw)
    except Exception:
        return None




def load_resume_state(path: str | Path) -> dict[str, Any] | None:
    """Load pipeline resume state metadata if present."""
    resume_path = _resolve_dataset_dir(path) / _RESUME_STATE_NAME
    if not resume_path.exists():
        return None
    try:
        raw = resume_path.read_bytes()
        return _loads(raw)
    except Exception:
        return None


def list_shard_paths(path: str | Path) -> list[str]:
    """Return resolved shard or jsonl file paths for a dataset input."""
    return [str(p) for p in _resolve_jsonl_inputs(path)]



def iter_jsonl_lines(path: str | Path) -> list[bytes]:
    """Read raw non-empty JSONL lines in deterministic shard order."""
    lines: list[bytes] = []
    for file_path in _resolve_jsonl_inputs(path):
        with file_path.open("rb") as f:
            for line in f:
                raw = line.rstrip(b"\n")
                if raw.strip():
                    lines.append(raw)
    return lines

def read_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    """Read canonical JSONL records as raw dictionaries.

    Accepts either:
    - a single jsonl file path
    - a directory path containing shard files
    - a missing base path that has matching `<stem>-*.jsonl` shard files
    """
    records: list[dict[str, Any]] = []
    for file_path in _resolve_jsonl_inputs(path):
        with file_path.open("rb") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(_loads(line))
    return records


def get_shard_stats(path: str | Path) -> list[dict[str, Any]]:
    """Return per-shard counts for dataset inspection."""
    stats: list[dict[str, Any]] = []
    for file_path in _resolve_jsonl_inputs(path):
        count = 0
        with file_path.open("rb") as f:
            for line in f:
                if line.strip():
                    count += 1
        stats.append({"path": str(file_path), "record_count": count})
    return stats




def summarize_dataset(path: str | Path) -> dict[str, Any]:
    """Return lightweight dataset summary for inspection/compare utilities."""
    records = read_jsonl_records(path)
    from collections import Counter

    stage_counts = Counter(str(r.get("stage_name", "missing")) for r in records)
    teacher_counts = Counter(str(r.get("teacher_name", "missing")) for r in records)

    avg_chunk = 0.0
    if records:
        avg_chunk = sum(int(r.get("byte_end", 0)) - int(r.get("byte_start", 0)) for r in records) / len(records)

    entropy_vals = [float(r["entropy"]) for r in records if r.get("entropy") is not None]
    avg_entropy = (sum(entropy_vals) / len(entropy_vals)) if entropy_vals else None

    return {
        "record_count": len(records),
        "stage_counts": dict(stage_counts),
        "teacher_counts": dict(teacher_counts),
        "avg_chunk_length": avg_chunk,
        "avg_entropy": avg_entropy,
        "schema_versions": sorted({str(r.get("schema_version", "missing")) for r in records}),
        "shard_count": len(get_shard_stats(path)),
        "manifest": load_dataset_manifest(path) or {},
    }



def sample_jsonl_records(
    path: str | Path,
    n: int,
    *,
    stage_name: str | None = None,
    teacher_name: str | None = None,
) -> list[dict[str, Any]]:
    """Return up to N deterministic records for manual inspection.

    Records can be filtered by stage/teacher and are selected in deterministic
    canonical order by `(doc_id, chunk_index, byte_start, byte_end)`.
    """
    if n <= 0:
        return []

    records = read_jsonl_records(path)
    filtered: list[dict[str, Any]] = []
    for rec in records:
        if stage_name is not None and str(rec.get("stage_name", "")) != stage_name:
            continue
        if teacher_name is not None and str(rec.get("teacher_name", "")) != teacher_name:
            continue
        filtered.append(rec)

    ordered = sorted(
        filtered,
        key=lambda r: (
            str(r.get("doc_id", "")),
            int(r.get("chunk_index", -1)) if r.get("chunk_index") is not None else -1,
            int(r.get("byte_start", -1)) if r.get("byte_start") is not None else -1,
            int(r.get("byte_end", -1)) if r.get("byte_end") is not None else -1,
        ),
    )
    return ordered[:n]

def read_jsonl(path: str | Path) -> list[DistilledSample]:
    """Read canonical distilled records from JSONL.

    Decodes compact top-k representations via `from_record()`.
    """
    return [from_record(r) for r in read_jsonl_records(path)]


def merge_jsonl_records(paths: list[str | Path]) -> list[dict[str, Any]]:
    """Merge multiple JSONL datasets by simple concatenation."""
    merged: list[dict[str, Any]] = []
    for path in paths:
        merged.extend(read_jsonl_records(path))
    return merged


def merge_jsonl_records_with_ratios(
    path_ratios: list[tuple[str | Path, float]],
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Merge JSONL datasets with deterministic ratio-based subsampling.

    This is a static/offline mixing helper for multi-teacher dataset composition.
    """
    from random import Random

    if not path_ratios:
        return []

    datasets: list[list[dict[str, Any]]] = []
    ratios: list[float] = []
    for path, ratio in path_ratios:
        datasets.append(read_jsonl_records(path))
        ratios.append(float(ratio))

    total_available = sum(len(ds) for ds in datasets)
    if total_available == 0:
        return []

    ratio_sum = sum(r for r in ratios if r > 0)
    if ratio_sum <= 0:
        return []

    targets = [int(total_available * (max(r, 0.0) / ratio_sum)) for r in ratios]
    remainder = total_available - sum(targets)
    fracs = sorted(
        [((total_available * (max(r, 0.0) / ratio_sum)) - targets[i], i) for i, r in enumerate(ratios)],
        reverse=True,
    )
    for _, idx in fracs[:remainder]:
        targets[idx] += 1

    mixed: list[dict[str, Any]] = []
    for i, ds in enumerate(datasets):
        if not ds or targets[i] <= 0:
            continue
        if targets[i] >= len(ds):
            mixed.extend(ds)
            continue
        rng = Random(seed + i + 1)
        idxs = list(range(len(ds)))
        rng.shuffle(idxs)
        chosen = sorted(idxs[: targets[i]])
        mixed.extend(ds[j] for j in chosen)

    return mixed
