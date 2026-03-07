#!/usr/bin/env python3
"""Compare two distillation dataset outputs in a terminal-friendly format."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from distill_factory.storage.reader import summarize_dataset


def _summary(path: str) -> dict[str, Any]:
    summary = summarize_dataset(path)
    summary["path"] = path
    return summary


def _print_dict_diff(title: str, left: dict[str, Any], right: dict[str, Any]) -> bool:
    changed = False
    keys = sorted(set(left.keys()) | set(right.keys()))
    print(title)
    if not keys:
        print("  (none)")
        return False

    for key in keys:
        lv = left.get(key, 0)
        rv = right.get(key, 0)
        if lv == rv:
            print(f"  {key}: {lv}")
        else:
            changed = True
            print(f"  DIFF {key}: left={lv} right={rv}")
    return changed


def _manifest_differences(left: dict[str, Any], right: dict[str, Any]) -> list[str]:
    ignore_keys = {"updated_timestamp", "creation_timestamp", "output_files"}
    keys = sorted((set(left.keys()) | set(right.keys())) - ignore_keys)
    diffs: list[str] = []
    for k in keys:
        if left.get(k) != right.get(k):
            diffs.append(k)
    return diffs


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two distillation datasets or runs.")
    parser.add_argument("--left", required=True, help="Path to left dataset file or directory")
    parser.add_argument("--right", required=True, help="Path to right dataset file or directory")
    args = parser.parse_args()

    left = _summary(args.left)
    right = _summary(args.right)

    print("=== Dataset Comparison ===")
    print(f"Left:  {left['path']}")
    print(f"Right: {right['path']}")

    print("\nConcise summary")
    for key in ["record_count", "shard_count"]:
        lv = left[key]
        rv = right[key]
        marker = "=" if lv == rv else "DIFF"
        print(f"  {marker} {key}: left={lv} right={rv}")

    lv = left["avg_chunk_length"]
    rv = right["avg_chunk_length"]
    marker = "=" if abs(lv - rv) < 1e-9 else "DIFF"
    print(f"  {marker} avg_chunk_length: left={lv:.2f} right={rv:.2f}")

    l_ent = left["avg_entropy"]
    r_ent = right["avg_entropy"]
    if l_ent is None and r_ent is None:
        print("  = avg_entropy: unavailable in both")
    elif l_ent is None or r_ent is None:
        print(f"  DIFF avg_entropy: left={l_ent} right={r_ent}")
    else:
        marker = "=" if abs(l_ent - r_ent) < 1e-9 else "DIFF"
        print(f"  {marker} avg_entropy: left={l_ent:.6f} right={r_ent:.6f}")

    left_schema = left["schema_versions"]
    right_schema = right["schema_versions"]
    schema_diff = left_schema != right_schema
    print(f"  {'DIFF' if schema_diff else '='} schema_versions: left={left_schema} right={right_schema}")

    print()
    stage_changed = _print_dict_diff("Record count by stage", left["stage_counts"], right["stage_counts"])
    print()
    teacher_changed = _print_dict_diff("Record count by teacher", left["teacher_counts"], right["teacher_counts"])

    manifest_diffs = _manifest_differences(left["manifest"], right["manifest"])
    print("\nManifest metadata differences")
    if not manifest_diffs:
        print("  (none)")
    else:
        for key in manifest_diffs:
            print(f"  DIFF {key}: left={left['manifest'].get(key)} right={right['manifest'].get(key)}")

    changed = any(
        [
            left["record_count"] != right["record_count"],
            left["shard_count"] != right["shard_count"],
            abs(left["avg_chunk_length"] - right["avg_chunk_length"]) >= 1e-9,
            left["avg_entropy"] != right["avg_entropy"],
            schema_diff,
            stage_changed,
            teacher_changed,
            bool(manifest_diffs),
        ]
    )
    print("\nOverall result")
    print(f"  {'DIFFERENT' if changed else 'MATCH'}")


if __name__ == "__main__":
    main()
