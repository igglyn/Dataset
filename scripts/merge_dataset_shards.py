#!/usr/bin/env python3
"""Merge dataset shards into a consolidated dataset layout."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from distill_factory.storage.reader import load_dataset_manifest

_SHARD_RE = re.compile(r"^(?P<prefix>.+)-(?P<idx>\d{5})\.jsonl$")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _group_input_files(input_dir: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for path in sorted(input_dir.glob("*.jsonl")):
        m = _SHARD_RE.match(path.name)
        key = m.group("prefix") if m else path.stem
        groups.setdefault(key, []).append(path)
    for key in list(groups.keys()):
        groups[key] = sorted(groups[key])
    return groups


def _read_nonempty_lines(paths: list[Path]) -> list[bytes]:
    lines: list[bytes] = []
    for p in paths:
        with p.open("rb") as f:
            for line in f:
                raw = line.rstrip(b"\n")
                if raw.strip():
                    lines.append(raw)
    return lines


def _write_lines(path: Path, lines: list[bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        for line in lines:
            f.write(line)
            f.write(b"\n")


def _write_group(
    output_dir: Path,
    group_name: str,
    lines: list[bytes],
    max_records_per_shard: int,
    shard_prefix: str,
) -> list[Path]:
    if max_records_per_shard <= 0:
        out_path = output_dir / f"{group_name}.jsonl"
        _write_lines(out_path, lines)
        return [out_path]

    written: list[Path] = []
    for idx, start in enumerate(range(0, len(lines), max_records_per_shard)):
        chunk = lines[start : start + max_records_per_shard]
        out = output_dir / f"{group_name}-{shard_prefix}-{idx:05d}.jsonl"
        _write_lines(out, chunk)
        written.append(out)
    if not written:
        out = output_dir / f"{group_name}-{shard_prefix}-00000.jsonl"
        _write_lines(out, [])
        written.append(out)
    return written


def _build_manifest(
    *,
    input_dir: Path,
    output_dir: Path,
    written_paths: list[Path],
    total_record_count: int,
    max_records_per_shard: int,
    shard_prefix: str,
) -> dict[str, Any]:
    src_manifest = load_dataset_manifest(input_dir)
    now = _now_iso()

    output_files: dict[str, Any] = {}
    for path in sorted(written_paths):
        count = 0
        with path.open("rb") as f:
            for line in f:
                if line.strip():
                    count += 1
        output_files[path.name] = {"path": str(path), "record_count": count}

    if isinstance(src_manifest, dict):
        manifest = dict(src_manifest)
    else:
        manifest = {}

    manifest["schema_version"] = manifest.get("schema_version", "1")
    manifest["creation_timestamp"] = manifest.get("creation_timestamp", now)
    manifest["updated_timestamp"] = now
    manifest["format"] = "jsonl"
    manifest["compression"] = "none"
    manifest["shard_count"] = len(written_paths)
    manifest["total_record_count"] = int(total_record_count)
    manifest["output_files"] = output_files
    manifest["format_settings"] = {
        "max_records_per_shard": int(max_records_per_shard),
        "shard_prefix": shard_prefix,
    }
    manifest["merged_from"] = str(input_dir)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge dataset shards into a consolidated output layout.")
    parser.add_argument("--input-dir", required=True, help="Input dataset directory containing JSONL shards/files.")
    parser.add_argument("--output-dir", required=True, help="Output directory for merged dataset.")
    parser.add_argument(
        "--max-records-per-shard",
        type=int,
        default=0,
        help="Optional reshaping target shard size (0 keeps one file per group).",
    )
    parser.add_argument("--shard-prefix", default="shard", help="Prefix used when resharing output files.")
    parser.add_argument("--copy-metrics", action="store_true", help="Copy teacher_quality_metrics.json if present.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        raise ValueError(f"--input-dir must be a directory: {input_dir}")

    groups = _group_input_files(input_dir)
    if not groups:
        raise ValueError(f"No JSONL files found in {input_dir}")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_written: list[Path] = []
    total_count = 0

    for group_name in sorted(groups.keys()):
        lines = _read_nonempty_lines(groups[group_name])
        total_count += len(lines)
        written = _write_group(
            output_dir=output_dir,
            group_name=group_name,
            lines=lines,
            max_records_per_shard=int(args.max_records_per_shard),
            shard_prefix=str(args.shard_prefix),
        )
        all_written.extend(written)
        print(f"{group_name}: {len(lines)} records -> {len(written)} file(s)")

    manifest = _build_manifest(
        input_dir=input_dir,
        output_dir=output_dir,
        written_paths=all_written,
        total_record_count=total_count,
        max_records_per_shard=int(args.max_records_per_shard),
        shard_prefix=str(args.shard_prefix),
    )
    (output_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    if args.copy_metrics:
        src_metrics = input_dir / "teacher_quality_metrics.json"
        if src_metrics.exists():
            shutil.copy2(src_metrics, output_dir / src_metrics.name)

    print(f"merged_record_count: {total_count}")
    print(f"manifest: {output_dir / 'dataset_manifest.json'}")


if __name__ == "__main__":
    main()
