"""Document-aware byte chunking utilities."""

from __future__ import annotations

from typing import Any


def chunk_document_bytes(
    doc: dict,
    chunk_bytes: int,
    overlap_bytes: int = 0,
    encoding: str = "utf-8",
) -> list[dict]:
    """Split one document into overlapping byte chunks."""
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    if overlap_bytes < 0 or overlap_bytes >= chunk_bytes:
        raise ValueError("overlap_bytes must be >= 0 and < chunk_bytes")

    payload = str(doc["text"]).encode(encoding)
    if not payload:
        return []

    step = chunk_bytes - overlap_bytes
    starts = list(range(0, len(payload), step))

    chunks: list[dict] = []
    total = len(starts)
    for i, start in enumerate(starts):
        end = min(start + chunk_bytes, len(payload))
        chunks.append(
            {
                "doc_id": doc["doc_id"],
                "chunk_index": i,
                "prev_chunk_index": (i - 1) if i > 0 else None,
                "next_chunk_index": (i + 1) if i < total - 1 else None,
                "byte_start": start,
                "byte_end": end,
                "raw_bytes": payload[start:end],
            }
        )
    return chunks


def chunk_documents(
    documents: list[dict],
    chunk_bytes: int,
    overlap_bytes: int = 0,
    encoding: str = "utf-8",
) -> list[dict]:
    """Chunk multiple documents without crossing boundaries."""
    out: list[dict] = []
    for doc in documents:
        out.extend(
            chunk_document_bytes(
                doc=doc,
                chunk_bytes=chunk_bytes,
                overlap_bytes=overlap_bytes,
                encoding=encoding,
            )
        )
    return out


def _reconstruct_doc_bytes(doc_records: list[dict[str, Any]]) -> bytes:
    if not doc_records:
        return b""
    max_end = max(int(r["byte_end"]) for r in doc_records)
    payload = bytearray(max_end)
    for r in doc_records:
        start = int(r["byte_start"])
        end = int(r["byte_end"])
        payload[start:end] = bytes(r["raw_bytes"])
    return bytes(payload)


def _select_window_bounds(
    target_start: int,
    target_end: int,
    neighborhood_start: int,
    neighborhood_end: int,
    context_window: int,
) -> tuple[int, int]:
    """Select a window around target, constrained by neighborhood and context limit."""
    max_size = max(1, context_window)
    total = neighborhood_end - neighborhood_start
    if total <= max_size:
        return neighborhood_start, neighborhood_end

    target_len = max(1, target_end - target_start)
    desired = max(max_size, target_len)

    centered_start = target_start - (desired - target_len) // 2
    window_start = max(neighborhood_start, centered_start)
    window_end = window_start + desired
    if window_end > neighborhood_end:
        window_end = neighborhood_end
        window_start = window_end - desired

    if target_start < window_start:
        window_start = target_start
        window_end = window_start + desired
    if target_end > window_end:
        window_end = target_end
        window_start = window_end - desired

    window_start = max(neighborhood_start, window_start)
    window_end = min(neighborhood_end, window_end)
    return window_start, window_end


def build_long_context_records(
    records: list[dict[str, Any]],
    context_window: int,
    stride: int,
) -> list[dict[str, Any]]:
    """Build per-target long-context windows from adjacent chunks in the same doc."""
    if context_window <= 0:
        raise ValueError("context_window must be positive")
    if stride < 0:
        raise ValueError("stride must be >= 0")

    by_doc: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        by_doc.setdefault(str(rec["doc_id"]), []).append(rec)

    doc_bytes: dict[str, bytes] = {}
    doc_ordered: dict[str, list[dict[str, Any]]] = {}
    for doc_id, doc_records in by_doc.items():
        ordered = sorted(doc_records, key=lambda r: int(r["chunk_index"]))
        doc_ordered[doc_id] = ordered
        doc_bytes[doc_id] = _reconstruct_doc_bytes(ordered)

    views: list[dict[str, Any]] = []
    for rec in records:
        doc_id = str(rec["doc_id"])
        ordered = doc_ordered[doc_id]
        payload = doc_bytes[doc_id]

        idx = int(rec["chunk_index"])
        left_idx = max(0, idx - stride)
        right_idx = min(len(ordered) - 1, idx + stride)
        neighborhood_start = int(ordered[left_idx]["byte_start"])
        neighborhood_end = int(ordered[right_idx]["byte_end"])

        target_start = int(rec["byte_start"])
        target_end = int(rec["byte_end"])
        window_start, window_end = _select_window_bounds(
            target_start=target_start,
            target_end=target_end,
            neighborhood_start=neighborhood_start,
            neighborhood_end=neighborhood_end,
            context_window=context_window,
        )
        window_bytes = payload[window_start:window_end]

        views.append(
            {
                "doc_id": doc_id,
                "chunk_index": idx,
                "window_byte_start": window_start,
                "window_byte_end": window_end,
                "window_raw_bytes": window_bytes,
                "target_byte_start_in_window": target_start - window_start,
                "target_byte_end_in_window": target_end - window_start,
            }
        )

    return views


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 0) -> list[str]:
    """Backward-compatible text chunking helper."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    payload = text.encode("utf-8")
    if not payload:
        return []

    step = chunk_size - overlap
    chunks: list[str] = []
    for start in range(0, len(payload), step):
        piece = payload[start : start + chunk_size]
        if piece:
            chunks.append(piece.decode("utf-8", errors="ignore"))
    return chunks
