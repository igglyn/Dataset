"""Reusable long-context helpers for stage B teacher inference."""

from __future__ import annotations

from typing import Any


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def _allocate_window_start(
    total_len: int,
    target_start: int,
    target_end: int,
    max_context: int,
    window_policy: str,
) -> int:
    target_len = max(0, target_end - target_start)
    max_start = max(0, total_len - max_context)

    if window_policy == "left_biased":
        return _clamp(target_end - max_context, 0, max_start)

    if window_policy == "right_biased":
        return _clamp(target_start, 0, max_start)

    target_mid = target_start + (target_len // 2)
    return _clamp(target_mid - (max_context // 2), 0, max_start)


def truncate_long_context_window(
    *,
    window_raw_bytes: bytes,
    target_start_offset: int,
    target_end_offset: int,
    max_teacher_context: int,
    window_policy: str = "center_target",
    target_region_policy: str = "preserve_full",
) -> dict[str, Any]:
    """Create teacher input window with explicit target-span metadata."""
    data = bytes(window_raw_bytes)
    n = len(data)

    if n == 0:
        return {
            "teacher_input_bytes": b"",
            "target_start_offset": 0,
            "target_end_offset": 0,
            "truncation_metadata": {
                "truncated": False,
                "window_policy": window_policy,
                "target_region_policy": target_region_policy,
                "original_window_bytes": 0,
                "final_window_bytes": 0,
            },
        }

    start = _clamp(int(target_start_offset), 0, n)
    end = _clamp(int(target_end_offset), start, n)
    target_len = end - start
    max_ctx = max(1, int(max_teacher_context))

    if max_ctx >= n:
        return {
            "teacher_input_bytes": data,
            "target_start_offset": start,
            "target_end_offset": end,
            "truncation_metadata": {
                "truncated": False,
                "window_policy": window_policy,
                "target_region_policy": target_region_policy,
                "original_window_bytes": n,
                "final_window_bytes": n,
            },
        }

    effective_policy = window_policy if window_policy in {"center_target", "left_biased", "right_biased"} else "center_target"
    effective_target_policy = (
        target_region_policy if target_region_policy in {"preserve_full", "truncate_if_needed"} else "preserve_full"
    )

    if target_len > max_ctx and effective_target_policy == "preserve_full":
        window_start = start
        window_end = end
    elif target_len > max_ctx and effective_target_policy == "truncate_if_needed":
        if effective_policy == "right_biased":
            kept_start = start
        elif effective_policy == "left_biased":
            kept_start = end - max_ctx
        else:
            kept_start = start + ((target_len - max_ctx) // 2)
        kept_start = _clamp(kept_start, start, end - max_ctx)
        kept_end = kept_start + max_ctx
        window_start = kept_start
        window_end = kept_end
    else:
        window_start = _allocate_window_start(n, start, end, max_ctx, effective_policy)
        window_end = min(n, window_start + max_ctx)

    final_bytes = data[window_start:window_end]
    final_target_start = _clamp(start - window_start, 0, len(final_bytes))
    final_target_end = _clamp(end - window_start, final_target_start, len(final_bytes))

    return {
        "teacher_input_bytes": final_bytes,
        "target_start_offset": final_target_start,
        "target_end_offset": final_target_end,
        "truncation_metadata": {
            "truncated": (window_start != 0 or window_end != n),
            "window_policy": effective_policy,
            "target_region_policy": effective_target_policy,
            "original_window_bytes": n,
            "final_window_bytes": len(final_bytes),
            "window_start_offset": window_start,
            "window_end_offset": window_end,
            "original_target_start_offset": start,
            "original_target_end_offset": end,
        },
    }


def prepare_long_context_teacher_input(
    *,
    window_raw_bytes: bytes,
    target_start_offset: int,
    target_end_offset: int,
    max_teacher_context: int,
    window_policy: str,
    target_region_policy: str,
    encoding: str = "utf-8",
) -> dict[str, Any]:
    """Build final teacher input (bytes + text) with explicit target-span metadata."""
    truncated = truncate_long_context_window(
        window_raw_bytes=window_raw_bytes,
        target_start_offset=target_start_offset,
        target_end_offset=target_end_offset,
        max_teacher_context=max_teacher_context,
        window_policy=window_policy,
        target_region_policy=target_region_policy,
    )
    teacher_bytes = truncated["teacher_input_bytes"]
    return {
        "teacher_input_bytes": teacher_bytes,
        "teacher_input_text": teacher_bytes.decode(encoding, errors="replace"),
        "target_start_offset": int(truncated["target_start_offset"]),
        "target_end_offset": int(truncated["target_end_offset"]),
        "truncation_metadata": dict(truncated["truncation_metadata"]),
    }
