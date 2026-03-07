"""Canonical distilled dataset record format helpers.

raw_bytes encoding strategy:
- `raw_bytes` is serialized as base64 text in field `raw_bytes_b64` for a stable,
  JSON-safe wire format across binary and UTF-8 content.
- the in-memory canonical record keeps `raw_bytes` as `bytes`.

Top-k compact storage strategy for JSONL:
- `top_k_ids` and `top_k_logprobs` keep their external field names but may be stored
  in compact encoded form as dictionaries:
  {
    "encoding": "ndarray_b64",
    "dtype": "<compact dtype>",
    "shape": [...],
    "data_b64": "..."
  }
- `top_k_ids` prefers smallest unsigned integer dtype: uint8 -> uint16 -> uint32.
- `top_k_logprobs` are packed as float16 when shape is regular.
- Reader remains backward compatible with legacy list-based fields.
- `hidden_summary` uses the same compact ndarray_b64 packing as float vectors (float16).
"""

from __future__ import annotations

import base64
import struct
from dataclasses import dataclass
from typing import Any

SCHEMA_VERSION = "1.0"


@dataclass(slots=True)
class DistilledSample:
    doc_id: str
    chunk_index: int
    byte_start: int
    byte_end: int
    raw_bytes: bytes
    split: str
    teacher_name: str
    stage_name: str
    mode: str
    target_doc_id: str | None = None
    target_chunk_index: int | None = None
    target_byte_start: int | None = None
    target_byte_end: int | None = None
    teacher_window_byte_start: int | None = None
    teacher_window_byte_end: int | None = None
    target_start_offset_within_window: int | None = None
    target_end_offset_within_window: int | None = None
    top_k_ids: list[int] | list[list[int]] | None = None
    top_k_logprobs: list[float] | list[list[float]] | None = None
    entropy: float | None = None
    hidden_summary: list[float] | list[list[float]] | None = None
    structured_output: dict[str, Any] | None = None
    extra_metadata: dict[str, Any] | None = None
    schema_version: str = SCHEMA_VERSION


_REQUIRED_FIELDS = {
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
}


def _is_rectangular_2d(values: list[Any]) -> bool:
    if not values:
        return True
    if not all(isinstance(row, list) for row in values):
        return False
    width = len(values[0])
    return all(len(row) == width for row in values)


def _flatten(values: list[Any]) -> tuple[list[float | int], list[int]]:
    if not values:
        return [], [0]
    if isinstance(values[0], list):
        if not _is_rectangular_2d(values):
            raise ValueError("non-rectangular")
        rows = len(values)
        cols = len(values[0])
        flat = [x for row in values for x in row]
        return flat, [rows, cols]
    return list(values), [len(values)]


def _unflatten(flat: list[float | int], shape: list[int]) -> list[Any]:
    if len(shape) == 1:
        return list(flat)
    rows, cols = shape
    out: list[list[float | int]] = []
    idx = 0
    for _ in range(rows):
        out.append(list(flat[idx : idx + cols]))
        idx += cols
    return out


def _pack_ids(values: list[Any]) -> dict[str, Any] | list[Any]:
    try:
        flat, shape = _flatten(values)
    except ValueError:
        return values

    if not flat:
        dtype = "uint8"
    else:
        max_v = max(int(v) for v in flat)
        if max_v <= 255:
            dtype = "uint8"
        elif max_v <= 65535:
            dtype = "uint16"
        else:
            dtype = "uint32"

    fmt = {"uint8": "B", "uint16": "H", "uint32": "I"}[dtype]
    packed = struct.pack(f"<{len(flat)}{fmt}", *[int(v) for v in flat]) if flat else b""
    return {
        "encoding": "ndarray_b64",
        "dtype": dtype,
        "shape": shape,
        "data_b64": base64.b64encode(packed).decode("ascii"),
    }


def _pack_logprobs(values: list[Any]) -> dict[str, Any] | list[Any]:
    try:
        flat, shape = _flatten(values)
    except ValueError:
        return values

    # float16 via struct format 'e'
    packed = struct.pack(f"<{len(flat)}e", *[float(v) for v in flat]) if flat else b""
    return {
        "encoding": "ndarray_b64",
        "dtype": "float16",
        "shape": shape,
        "data_b64": base64.b64encode(packed).decode("ascii"),
    }




def _pack_hidden_summary(values: list[Any]) -> dict[str, Any] | list[Any]:
    try:
        flat, shape = _flatten(values)
    except ValueError:
        return values

    packed = struct.pack(f"<{len(flat)}e", *[float(v) for v in flat]) if flat else b""
    return {
        "encoding": "ndarray_b64",
        "dtype": "float16",
        "shape": shape,
        "data_b64": base64.b64encode(packed).decode("ascii"),
    }


def _decode_compact(encoded: Any) -> Any:
    if not isinstance(encoded, dict) or encoded.get("encoding") != "ndarray_b64":
        return encoded

    dtype = str(encoded["dtype"])
    shape = [int(v) for v in encoded["shape"]]
    raw = base64.b64decode(str(encoded["data_b64"]).encode("ascii"))

    if dtype == "uint8":
        unit = "B"
    elif dtype == "uint16":
        unit = "H"
    elif dtype == "uint32":
        unit = "I"
    elif dtype == "float16":
        unit = "e"
    else:
        raise ValueError(f"Unsupported compact dtype: {dtype}")

    count = 1
    for s in shape:
        count *= s
    flat = list(struct.unpack(f"<{count}{unit}", raw)) if count > 0 else []

    if dtype.startswith("uint"):
        flat = [int(v) for v in flat]
    else:
        flat = [float(v) for v in flat]

    return _unflatten(flat, shape)


def _cast_float16_scalar(value: float) -> float:
    return float(struct.unpack("<e", struct.pack("<e", float(value)))[0])


def to_record(sample: DistilledSample) -> dict[str, Any]:
    """Serialize an in-memory sample into canonical JSON-safe dict."""
    structured_output = sample.structured_output
    if structured_output is None and isinstance(sample.extra_metadata, dict):
        maybe = sample.extra_metadata.get("structured_output")
        if isinstance(maybe, dict):
            structured_output = maybe

    top_k_ids_wire = _pack_ids(sample.top_k_ids) if isinstance(sample.top_k_ids, list) else sample.top_k_ids
    top_k_logprobs_wire = (
        _pack_logprobs(sample.top_k_logprobs) if isinstance(sample.top_k_logprobs, list) else sample.top_k_logprobs
    )
    entropy_wire = None if sample.entropy is None else _cast_float16_scalar(sample.entropy)
    hidden_summary_wire = (
        _pack_hidden_summary(sample.hidden_summary) if isinstance(sample.hidden_summary, list) else sample.hidden_summary
    )

    return {
        "schema_version": sample.schema_version,
        "doc_id": sample.doc_id,
        "chunk_index": sample.chunk_index,
        "byte_start": sample.byte_start,
        "byte_end": sample.byte_end,
        "raw_bytes_b64": base64.b64encode(sample.raw_bytes).decode("ascii"),
        "target_doc_id": sample.target_doc_id,
        "target_chunk_index": sample.target_chunk_index,
        "target_byte_start": sample.target_byte_start,
        "target_byte_end": sample.target_byte_end,
        "teacher_window_byte_start": sample.teacher_window_byte_start,
        "teacher_window_byte_end": sample.teacher_window_byte_end,
        "target_start_offset_within_window": sample.target_start_offset_within_window,
        "target_end_offset_within_window": sample.target_end_offset_within_window,
        "split": sample.split,
        "teacher_name": sample.teacher_name,
        "stage_name": sample.stage_name,
        "mode": sample.mode,
        "top_k_ids": top_k_ids_wire,
        "top_k_logprobs": top_k_logprobs_wire,
        "entropy": entropy_wire,
        "hidden_summary": hidden_summary_wire,
        "structured_output": structured_output,
        "extra_metadata": sample.extra_metadata,
    }


def from_record(record: dict[str, Any]) -> DistilledSample:
    """Deserialize canonical dict into in-memory sample with raw bytes."""
    missing = sorted(_REQUIRED_FIELDS - set(record))
    if missing:
        raise ValueError(f"Record missing required fields: {missing}")
    if str(record["schema_version"]) != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema_version: {record['schema_version']}")

    raw_value = record["raw_bytes_b64"]
    raw_bytes = base64.b64decode(raw_value.encode("ascii"))

    extra_metadata = record.get("extra_metadata")
    if extra_metadata is not None and not isinstance(extra_metadata, dict):
        raise ValueError("extra_metadata must be a dict or null")

    structured_output = record.get("structured_output")
    if structured_output is not None and not isinstance(structured_output, dict):
        raise ValueError("structured_output must be a dict or null")

    top_k_ids = _decode_compact(record.get("top_k_ids"))
    top_k_logprobs = _decode_compact(record.get("top_k_logprobs"))
    hidden_summary = _decode_compact(record.get("hidden_summary"))

    return DistilledSample(
        doc_id=str(record["doc_id"]),
        chunk_index=int(record["chunk_index"]),
        byte_start=int(record["byte_start"]),
        byte_end=int(record["byte_end"]),
        raw_bytes=raw_bytes,
        target_doc_id=None if record.get("target_doc_id") is None else str(record.get("target_doc_id")),
        target_chunk_index=None if record.get("target_chunk_index") is None else int(record.get("target_chunk_index")),
        target_byte_start=None if record.get("target_byte_start") is None else int(record.get("target_byte_start")),
        target_byte_end=None if record.get("target_byte_end") is None else int(record.get("target_byte_end")),
        teacher_window_byte_start=None
        if record.get("teacher_window_byte_start") is None
        else int(record.get("teacher_window_byte_start")),
        teacher_window_byte_end=None if record.get("teacher_window_byte_end") is None else int(record.get("teacher_window_byte_end")),
        target_start_offset_within_window=None
        if record.get("target_start_offset_within_window") is None
        else int(record.get("target_start_offset_within_window")),
        target_end_offset_within_window=None
        if record.get("target_end_offset_within_window") is None
        else int(record.get("target_end_offset_within_window")),
        split=str(record["split"]),
        teacher_name=str(record["teacher_name"]),
        stage_name=str(record["stage_name"]),
        mode=str(record["mode"]),
        top_k_ids=top_k_ids,
        top_k_logprobs=top_k_logprobs,
        entropy=None if record.get("entropy") is None else float(record["entropy"]),
        hidden_summary=hidden_summary,
        structured_output=structured_output,
        extra_metadata=extra_metadata,
    )
