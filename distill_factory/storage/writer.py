"""Storage writing helpers."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any

from distill_factory.data.formats import DistilledSample, SCHEMA_VERSION, to_record
from distill_factory.utils.hashing import record_signature

try:
    import orjson
except ModuleNotFoundError:  # pragma: no cover
    orjson = None
    import json


_MANIFEST_NAME = "dataset_manifest.json"


def _dumps(record: dict[str, Any]) -> bytes:
    if orjson is not None:
        return orjson.dumps(record)
    return json.dumps(record, ensure_ascii=False).encode("utf-8")


def _loads(blob: bytes) -> dict[str, Any]:
    if orjson is not None:
        return orjson.loads(blob)
    return json.loads(blob.decode("utf-8"))


def _record_signature_from_sample(sample: DistilledSample) -> str:
    return record_signature(
        {
            "doc_id": sample.doc_id,
            "chunk_index": sample.chunk_index,
            "byte_start": sample.byte_start,
            "byte_end": sample.byte_end,
            "stage_name": sample.stage_name,
            "teacher_name": sample.teacher_name,
            "mode": sample.mode,
        }
    )


def _deduplicate_samples(records: list[DistilledSample]) -> tuple[list[DistilledSample], int]:
    """Drop duplicate records deterministically by signature before writing."""
    seen: set[str] = set()
    unique: list[DistilledSample] = []
    duplicates = 0
    for sample in records:
        sig = _record_signature_from_sample(sample)
        if sig in seen:
            duplicates += 1
            continue
        seen.add(sig)
        unique.append(sample)
    return unique, duplicates


def _write_jsonl_file(records: list[DistilledSample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        for sample in records:
            f.write(_dumps(to_record(sample)))
            f.write(b"\n")


def _append_jsonl_file(records: list[DistilledSample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as f:
        for sample in records:
            f.write(_dumps(to_record(sample)))
            f.write(b"\n")


def _read_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return _loads(path.read_bytes())
    except Exception:
        return None


def _count_nonempty_lines(path: Path) -> int:
    count = 0
    with path.open("rb") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _load_existing_signatures(paths: list[Path]) -> set[str]:
    seen: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open("rb") as f:
            for line in f:
                raw = line.strip()
                if not raw:
                    continue
                rec = _loads(raw)
                seen.add(
                    record_signature(
                        {
                            "doc_id": rec.get("doc_id"),
                            "chunk_index": rec.get("chunk_index"),
                            "byte_start": rec.get("byte_start"),
                            "byte_end": rec.get("byte_end"),
                            "stage_name": rec.get("stage_name"),
                            "teacher_name": rec.get("teacher_name"),
                            "mode": rec.get("mode"),
                        }
                    )
                )
    return seen


def _split_new_records(records: list[DistilledSample], existing_signatures: set[str]) -> list[DistilledSample]:
    out: list[DistilledSample] = []
    for sample in records:
        sig = _record_signature_from_sample(sample)
        if sig in existing_signatures:
            continue
        existing_signatures.add(sig)
        out.append(sample)
    return out


def _update_dataset_manifest(
    output_path: Path,
    written_paths: list[Path],
    unique_records: list[DistilledSample],
    max_records_per_shard: int,
    shard_prefix: str,
    skipped_records_count: int = 0,
) -> None:
    dataset_dir = output_path.parent
    manifest_path = dataset_dir / _MANIFEST_NAME
    now = datetime.now(timezone.utc).isoformat()

    existing = _read_manifest(manifest_path) or {}

    stage_names = sorted({s.stage_name for s in unique_records if s.stage_name})
    teacher_names = sorted({s.teacher_name for s in unique_records if s.teacher_name})

    output_file_info = {
        p.name: {"path": str(p), "record_count": _count_nonempty_lines(p)} for p in sorted(set(written_paths)) if p.exists()
    }

    existing_outputs = existing.get("output_files")
    if not isinstance(existing_outputs, dict):
        existing_outputs = {}
    existing_outputs.update(output_file_info)

    all_jsonl_files = sorted(dataset_dir.glob("*.jsonl"))
    total_record_count = sum(v.get("record_count", 0) for v in existing_outputs.values() if isinstance(v, dict))

    try:
        existing_skipped = int(existing.get("skipped_record_count", 0) or 0)
    except (TypeError, ValueError):
        existing_skipped = 0

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "creation_timestamp": existing.get("creation_timestamp", now),
        "updated_timestamp": now,
        "config_path": existing.get("config_path") or os.environ.get("DISTILL_FACTORY_CONFIG_PATH"),
        "config_snapshot": existing.get("config_snapshot") or os.environ.get("DISTILL_FACTORY_CONFIG_SNAPSHOT"),
        "enabled_stages": sorted(set(existing.get("enabled_stages", [])) | set(stage_names)),
        "teacher_names": sorted(set(existing.get("teacher_names", [])) | set(teacher_names)),
        "shard_count": len(all_jsonl_files),
        "total_record_count": int(total_record_count) if total_record_count >= 0 else None,
        "format": "jsonl",
        "compression": "none",
        "format_settings": {
            "max_records_per_shard": int(max_records_per_shard),
            "shard_prefix": shard_prefix,
        },
        "output_files": existing_outputs,
        "skipped_record_count": int(existing_skipped + max(0, int(skipped_records_count))),
    }

    manifest_path.write_bytes(_dumps(manifest))


def _resolve_shard_paths(base_path: Path, shard_prefix: str) -> list[Path]:
    return sorted(base_path.parent.glob(f"{shard_prefix}-*.jsonl"))


def write_jsonl(
    records: list[DistilledSample],
    path: str | Path,
    max_records_per_shard: int = 0,
    shard_prefix: str = "shard",
    append: bool = False,
    deduplicate: bool = True,
    skipped_records_count: int = 0,
) -> list[str]:
    """Write canonical distilled records as JSONL.

    - Non-sharded mode (default): writes a single file at `path`.
    - Sharded mode: when `max_records_per_shard > 0` and record count exceeds it,
      writes deterministic shard files in `path.parent` using sortable names:
      `{shard_prefix}-00000.jsonl`, `{shard_prefix}-00001.jsonl`, ...

    If `append=True`, existing output is preserved and only new records are written.
    De-duplication is deterministic against both incoming and existing records.
    Set `deduplicate=False` to preserve incoming records exactly as provided.

    Returns the list of file paths touched by this write operation.
    """
    p = Path(path)
    if deduplicate:
        unique_records, _ = _deduplicate_samples(records)
    else:
        unique_records = list(records)

    if max_records_per_shard <= 0:
        existing_paths = [p] if append and p.exists() else []
        existing_signatures = _load_existing_signatures(existing_paths) if append else set()
        to_write = _split_new_records(unique_records, existing_signatures) if append else unique_records

        if append and p.exists():
            _append_jsonl_file(to_write, p)
        else:
            _write_jsonl_file(to_write, p)

        written = [p]
        _update_dataset_manifest(p, written, unique_records, max_records_per_shard, shard_prefix, skipped_records_count=skipped_records_count)
        return [str(p)]

    shard_paths = _resolve_shard_paths(p, shard_prefix)
    existing_signatures = _load_existing_signatures(shard_paths) if append else set()
    to_write = _split_new_records(unique_records, existing_signatures) if append else unique_records

    written: list[Path] = []
    if not append:
        for old in shard_paths:
            old.unlink(missing_ok=True)
        shard_paths = []

    if not shard_paths:
        shard_paths = []

    if append and shard_paths:
        last_path = shard_paths[-1]
        current_count = _count_nonempty_lines(last_path)
        remaining_capacity = max(0, max_records_per_shard - current_count)
        if remaining_capacity > 0 and to_write:
            tail = to_write[:remaining_capacity]
            _append_jsonl_file(tail, last_path)
            written.append(last_path)
            to_write = to_write[remaining_capacity:]

    next_idx = 0
    if shard_paths:
        try:
            next_idx = int(shard_paths[-1].stem.split("-")[-1]) + 1
        except Exception:
            next_idx = len(shard_paths)

    for start in range(0, len(to_write), max_records_per_shard):
        end = start + max_records_per_shard
        shard_path = p.parent / f"{shard_prefix}-{next_idx:05d}.jsonl"
        _write_jsonl_file(to_write[start:end], shard_path)
        written.append(shard_path)
        next_idx += 1

    if not written and shard_paths:
        written = shard_paths

    _update_dataset_manifest(p, written, unique_records, max_records_per_shard, shard_prefix, skipped_records_count=skipped_records_count)
    return [str(s) for s in written]


def write_parquet(records: list[DistilledSample], path: str | Path) -> None:
    """TODO: add pyarrow parquet persistence.

    Intended parquet policy (explicit):
    - top_k_ids column -> smallest fitting integer physical dtype
    - top_k_logprobs column -> float16/float32 depending on parquet engine support
    - entropy column -> float16/float32 scalar
    """
    raise NotImplementedError("Parquet writer is TODO. Use write_jsonl for now.")
