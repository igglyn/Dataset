"""Hashing helpers."""

from __future__ import annotations

import hashlib
from typing import Any


def sha256_text(text: str) -> str:
    """Return SHA256 hex digest for text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def record_identity_string(record: dict[str, Any]) -> str:
    """Build stable, human-readable identity string for deduplication.

    Signature includes required semantic identity fields:
    - doc_id
    - chunk_index (fallback to byte range if needed)
    - stage_name
    - teacher_name
    - mode
    """
    doc_id = str(record.get("doc_id", ""))
    stage_name = str(record.get("stage_name", ""))
    teacher_name = str(record.get("teacher_name", ""))
    mode = str(record.get("mode", ""))

    chunk_index = record.get("chunk_index")
    if chunk_index is None:
        byte_start = record.get("byte_start")
        byte_end = record.get("byte_end")
        chunk_part = f"bytes:{byte_start}-{byte_end}"
    else:
        chunk_part = f"chunk:{int(chunk_index)}"

    return "|".join([doc_id, chunk_part, stage_name, teacher_name, mode])


def record_signature(record: dict[str, Any]) -> str:
    """Return stable SHA256 signature for a record identity."""
    return sha256_text(record_identity_string(record))


def deduplicate_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Drop duplicate records deterministically by first-seen order.

    Returns `(unique_records, duplicates_skipped)`.
    """
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    duplicates = 0
    for record in records:
        sig = record_signature(record)
        if sig in seen:
            duplicates += 1
            continue
        seen.add(sig)
        unique.append(record)
    return unique, duplicates
