"""Pure utilities for position-level selection policy.

Semantics:
- Entropy threshold keeps positions where ``per_token_entropy[i] >= threshold``.
- Top1-gap threshold keeps positions where ``per_token_top1_gap[i] <= threshold``.
- Combined selection defaults to ``union`` (logical OR) unless explicitly set to
  ``intersection`` (logical AND).
"""

from __future__ import annotations

from typing import Iterable


def select_positions_by_entropy(per_token_entropy: list[float], threshold: float) -> list[bool]:
    """Return mask for positions with entropy >= threshold."""
    return [float(v) >= float(threshold) for v in per_token_entropy]



def select_positions_by_top1_gap(per_token_top1_gap: list[float], threshold: float) -> list[bool]:
    """Return mask for positions with top1 gap <= threshold."""
    return [float(v) <= float(threshold) for v in per_token_top1_gap]



def combine_position_masks(mask_a: list[bool], mask_b: list[bool], mode: str = "union") -> list[bool]:
    """Combine two masks with union (OR) or intersection (AND)."""
    if len(mask_a) != len(mask_b):
        raise ValueError("Position masks must have equal length")
    if mode == "union":
        return [bool(a) or bool(b) for a, b in zip(mask_a, mask_b)]
    if mode == "intersection":
        return [bool(a) and bool(b) for a, b in zip(mask_a, mask_b)]
    raise ValueError("mode must be 'union' or 'intersection'")



def enforce_minimum_selected_positions(mask: list[bool], minimum_selected_positions: int) -> list[bool]:
    """Ensure at least ``minimum_selected_positions`` positions are selected.

    If the mask already satisfies the minimum, it is returned as-is.
    Otherwise, additional positions are enabled from left to right.
    """
    minimum = int(minimum_selected_positions)
    if minimum <= 0 or not mask:
        return list(mask)

    out = [bool(v) for v in mask]
    selected = sum(1 for v in out if v)
    if selected >= minimum:
        return out

    need = minimum - selected
    for i, current in enumerate(out):
        if not current:
            out[i] = True
            need -= 1
            if need == 0:
                break
    return out



def mask_to_windows(mask: list[bool], radius: int) -> list[tuple[int, int]]:
    """Expand selected positions into inclusive windows and merge overlaps."""
    if radius < 0:
        raise ValueError("radius must be >= 0")
    n = len(mask)
    windows: list[tuple[int, int]] = []
    for i, selected in enumerate(mask):
        if not selected:
            continue
        start = max(0, i - radius)
        end = min(n - 1, i + radius)
        windows.append((start, end))
    return merge_overlapping_windows(windows)



def merge_overlapping_windows(windows: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping inclusive windows in sorted order."""
    normalized = sorted((int(s), int(e)) for s, e in windows)
    if not normalized:
        return []

    merged: list[list[int]] = [[normalized[0][0], normalized[0][1]]]
    for start, end in normalized[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]



def select_positions(
    *,
    per_token_entropy: list[float] | None,
    per_token_top1_gap: list[float] | None,
    entropy_threshold: float | None,
    top1_gap_threshold: float | None,
    combine_mode: str = "union",
    minimum_selected_positions: int | None = None,
) -> list[bool]:
    """Select informative positions from optional entropy/gap signals.

    Default combined selection is union (OR) when both thresholds are provided.
    """
    entropy_mask: list[bool] | None = None
    gap_mask: list[bool] | None = None

    if entropy_threshold is not None:
        if per_token_entropy is None:
            raise ValueError("per_token_entropy is required when entropy_threshold is set")
        entropy_mask = select_positions_by_entropy(per_token_entropy, entropy_threshold)

    if top1_gap_threshold is not None:
        if per_token_top1_gap is None:
            raise ValueError("per_token_top1_gap is required when top1_gap_threshold is set")
        gap_mask = select_positions_by_top1_gap(per_token_top1_gap, top1_gap_threshold)

    if entropy_mask is None and gap_mask is None:
        base: list[bool] = [False] * len(per_token_entropy or per_token_top1_gap or [])
    elif entropy_mask is None:
        base = gap_mask or []
    elif gap_mask is None:
        base = entropy_mask
    else:
        base = combine_position_masks(entropy_mask, gap_mask, mode=combine_mode)

    if minimum_selected_positions is not None:
        return enforce_minimum_selected_positions(base, minimum_selected_positions)
    return base
