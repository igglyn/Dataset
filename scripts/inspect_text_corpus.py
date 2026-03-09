#!/usr/bin/env python3
"""Inspect cached source datasets or built mixed corpora from the terminal."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from typing import Any

CANONICAL_SPLITS = ("train", "eval", "validation")


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _split_doc_counts(root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split in CANONICAL_SPLITS:
        split_dir = root / split
        if split_dir.is_dir():
            counts[split] = len(list(split_dir.glob("doc_*.txt")))
    return counts


def _count_by_metadata_field(root: Path, field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split in CANONICAL_SPLITS:
        split_dir = root / split
        if not split_dir.is_dir():
            continue
        for meta_path in sorted(split_dir.glob("doc_*.meta.json")):
            payload = _load_json(meta_path)
            if not payload:
                continue
            key_raw = payload.get(field)
            if key_raw is None:
                continue
            key = str(key_raw)
            counts[key] = counts.get(key, 0) + 1
    return counts


def _find_sample_doc(root: Path) -> tuple[Path, Path] | None:
    for split in CANONICAL_SPLITS:
        split_dir = root / split
        if not split_dir.is_dir():
            continue
        for txt in sorted(split_dir.glob("doc_*.txt")):
            meta = txt.with_suffix(".meta.json")
            return txt, meta
    return None


def _manifest_summary(manifest: dict[str, Any]) -> list[str]:
    keys_of_interest = [
        "mixture_name",
        "source_name",
        "target_documents",
        "target_documents_per_split",
        "random_seed",
        "created_at",
        "extraction_timestamp",
        "composition_deviation",
    ]
    lines: list[str] = []
    for key in keys_of_interest:
        if key in manifest:
            lines.append(f"  - {key}: {manifest[key]}")
    lines.append(f"  - manifest_keys: {', '.join(sorted(manifest.keys()))}")
    return lines


def _preview_text(path: Path, max_chars: int) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"


def inspect_corpus(path: Path, preview_chars: int = 400) -> str:
    if not path.is_dir():
        raise ValueError(f"Path is not a directory: {path}")

    lines: list[str] = []
    lines.append(f"Inspecting: {path}")

    manifest = _load_json(path / "manifest.json")
    if manifest is None:
        lines.append("Manifest: missing or unreadable")
    else:
        lines.append("Manifest summary:")
        lines.extend(_manifest_summary(manifest))

    split_counts = _split_doc_counts(path)
    lines.append("Split counts:")
    if split_counts:
        for split in CANONICAL_SPLITS:
            if split in split_counts:
                lines.append(f"  - {split}: {split_counts[split]}")
    else:
        lines.append("  - no canonical split directories found")

    by_source = _count_by_metadata_field(path, "source_name")
    lines.append("Document counts by source_name:")
    if by_source:
        for key in sorted(by_source):
            lines.append(f"  - {key}: {by_source[key]}")
    else:
        lines.append("  - unavailable (no source_name in sidecar metadata)")

    by_group = _count_by_metadata_field(path, "group_name")
    lines.append("Document counts by group_name:")
    if by_group:
        for key in sorted(by_group):
            lines.append(f"  - {key}: {by_group[key]}")
    else:
        lines.append("  - unavailable (no group_name in sidecar metadata)")

    sample = _find_sample_doc(path)
    if sample is None:
        lines.append("Sample preview: no documents found")
    else:
        txt_path, meta_path = sample
        lines.append(f"Sample text: {txt_path}")
        lines.append(_preview_text(txt_path, max_chars=preview_chars))
        lines.append(f"Sample metadata: {meta_path}")
        meta = _load_json(meta_path)
        if meta is None:
            lines.append("  - missing or unreadable metadata")
        else:
            lines.append(json.dumps(meta, indent=2, sort_keys=True))

    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Path to data/sources/<source_name>/ or data/corpora/<mixture_name>/")
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=400,
        help="Max characters to print for sample text preview (default: 400)",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.preview_chars <= 0:
        raise SystemExit("--preview-chars must be > 0")
    report = inspect_corpus(Path(args.path), preview_chars=args.preview_chars)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
