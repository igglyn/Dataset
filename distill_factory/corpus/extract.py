"""Deterministic source-dataset extraction into a persistent text-document cache layout."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any, Iterable

from distill_factory.corpus.manifest import (
    config_fingerprint,
    load_manifest,
    manifest_matches_source,
    source_config_payload,
    write_manifest,
)
from distill_factory.corpus.schema import SourceDatasetCacheConfig

CANONICAL_SPLITS = ("train", "eval", "validation")


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_hf_streaming_split(source: SourceDatasetCacheConfig, split_name: str) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise RuntimeError("huggingface 'datasets' package is required for source extraction") from exc

    try:
        return load_dataset(
            source.hf_dataset,
            source.hf_config,
            split=split_name,
            streaming=True,
        )
    except Exception as exc:  # clear failure surface for unavailable splits/datasets
        raise ValueError(
            f"Failed to load streaming split '{split_name}' for source '{source.source_name}' "
            f"({source.hf_dataset}/{source.hf_config})"
        ) from exc


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _count_existing_docs(split_dir: Path) -> int:
    return len(list(split_dir.glob("doc_*.meta.json")))


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _build_group_text(rows: list[dict[str, Any]], text_field: str) -> str:
    parts = [_safe_text(row.get(text_field, "")) for row in rows]
    return "\n\n".join(parts).strip()


def _byte_len_utf8(text: str) -> int:
    return len(text.encode("utf-8"))


def inspect_source_cache_state(source: SourceDatasetCacheConfig, cache_root: str | Path) -> dict[str, Any]:
    """Inspect whether cache is reusable, missing, or requires refresh."""
    source_dir = Path(cache_root) / source.source_name
    if not source_dir.exists():
        return {"state": "missing", "source_dir": str(source_dir)}

    manifest = load_manifest(source_dir)
    if manifest is None:
        return {
            "state": "present_without_manifest",
            "source_dir": str(source_dir),
            "reason": "cache exists but manifest.json is missing",
        }

    if manifest_matches_source(manifest, source):
        return {
            "state": "ready",
            "source_dir": str(source_dir),
            "manifest": manifest,
        }

    return {
        "state": "mismatch",
        "source_dir": str(source_dir),
        "reason": "existing cache fingerprint differs from requested config",
        "existing_config_fingerprint": manifest.get("config_fingerprint"),
        "requested_config_fingerprint": config_fingerprint(source),
    }


def _extract_split(
    source: SourceDatasetCacheConfig,
    canonical_split: str,
    upstream_split: str,
    split_dir: Path,
) -> dict[str, Any]:
    split_dir.mkdir(parents=True, exist_ok=True)
    existing_docs = _count_existing_docs(split_dir)
    accepted_rows_to_skip_for_resume = existing_docs * source.group_size

    stream = _load_hf_streaming_split(source=source, split_name=upstream_split)

    buffered_rows: list[dict[str, Any]] = []
    skipped_rows_for_resume = 0
    total_rows_seen = 0
    groups_written = 0
    filtered_below_min_bytes = 0
    filtered_above_max_bytes = 0

    max_docs = source.max_docs_per_split
    for row_ordinal, row in enumerate(stream):
        total_rows_seen = row_ordinal + 1

        row_text = _safe_text(row.get(source.text_field, ""))
        row_bytes = _byte_len_utf8(row_text)

        if source.min_bytes is not None and row_bytes < source.min_bytes:
            filtered_below_min_bytes += 1
            continue
        if source.max_bytes is not None and row_bytes > source.max_bytes:
            filtered_above_max_bytes += 1
            continue

        if skipped_rows_for_resume < accepted_rows_to_skip_for_resume:
            skipped_rows_for_resume += 1
            continue

        buffered_rows.append(row)
        if len(buffered_rows) < source.group_size:
            continue

        doc_index = existing_docs + groups_written + 1
        doc_stem = f"doc_{doc_index:08d}"

        text = _build_group_text(buffered_rows, source.text_field)
        extraction_timestamp = _utc_timestamp()
        meta = {
            "source_name": source.source_name,
            "source_type": source.source_type,
            "hf_dataset": source.hf_dataset,
            "hf_config": source.hf_config,
            "split": canonical_split,
            "upstream_split": upstream_split,
            "text_field": source.text_field,
            "group_size": source.group_size,
            "min_bytes": source.min_bytes,
            "max_bytes": source.max_bytes,
            "row_ordinal_start": row_ordinal - (source.group_size - 1),
            "row_ordinal_end": row_ordinal,
            "extraction_timestamp": extraction_timestamp,
        }

        (split_dir / f"{doc_stem}.txt").write_text(text + "\n", encoding="utf-8")
        _write_json(split_dir / f"{doc_stem}.meta.json", meta)

        groups_written += 1
        buffered_rows = []

        if max_docs is not None and (existing_docs + groups_written) >= max_docs:
            break

    return {
        "canonical_split": canonical_split,
        "upstream_split": upstream_split,
        "existing_docs_before": existing_docs,
        "docs_written": groups_written,
        "docs_total": existing_docs + groups_written,
        "rows_skipped_for_resume": skipped_rows_for_resume,
        "rows_seen": total_rows_seen,
        "group_size": source.group_size,
        "min_bytes": source.min_bytes,
        "max_bytes": source.max_bytes,
        "filtered_below_min_bytes": filtered_below_min_bytes,
        "filtered_above_max_bytes": filtered_above_max_bytes,
    }


def extract_source_to_cache(
    source: SourceDatasetCacheConfig,
    cache_root: str | Path,
    *,
    refresh: bool = False,
    dry_run: bool = False,
) -> Path:
    """Extract one configured source dataset into deterministic cached text-doc layout."""
    source_dir = Path(cache_root) / source.source_name
    state = inspect_source_cache_state(source=source, cache_root=cache_root)

    if state["state"] == "ready" and not refresh:
        if dry_run:
            return source_dir
        return source_dir

    if state["state"] == "mismatch" and not refresh:
        raise ValueError(
            "Existing cache config differs from requested source config. "
            "Re-run with refresh/overwrite to rebuild safely."
        )

    if state["state"] == "present_without_manifest" and not refresh:
        raise ValueError(
            "Existing cache directory is missing manifest.json. "
            "Re-run with refresh/overwrite to regenerate safely."
        )

    if dry_run:
        return source_dir

    if refresh and source_dir.exists():
        shutil.rmtree(source_dir)

    source_dir.mkdir(parents=True, exist_ok=True)

    split_dirs = {split: source_dir / split for split in CANONICAL_SPLITS}
    for split_dir in split_dirs.values():
        split_dir.mkdir(parents=True, exist_ok=True)

    split_results: list[dict[str, Any]] = []
    for canonical_split in CANONICAL_SPLITS:
        upstream_split = source.split_mapping.get(canonical_split)
        if upstream_split is None:
            continue
        split_results.append(
            _extract_split(
                source=source,
                canonical_split=canonical_split,
                upstream_split=upstream_split,
                split_dir=split_dirs[canonical_split],
            )
        )

    extracted_doc_counts = {
        split_result["canonical_split"]: split_result["docs_total"]
        for split_result in split_results
    }

    manifest = {
        "source_name": source.source_name,
        "source_type": source.source_type,
        "hf_dataset": source.hf_dataset,
        "hf_config": source.hf_config,
        "split_mapping": source.split_mapping,
        "text_field": source.text_field,
        "group_size": source.group_size,
        "max_docs_per_split": source.max_docs_per_split,
        "min_bytes": source.min_bytes,
        "max_bytes": source.max_bytes,
        "extracted_doc_counts": extracted_doc_counts,
        "extraction_timestamp": _utc_timestamp(),
        "config_fingerprint": config_fingerprint(source),
        "source_config": source_config_payload(source),
        "splits": split_results,
    }
    write_manifest(source_dir=source_dir, manifest=manifest)
    return source_dir
