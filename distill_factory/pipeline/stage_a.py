"""Stage A execution (bulk grounding teacher)."""

from __future__ import annotations

import os
from typing import Any

from distill_factory.data.selection import mask_to_windows, select_positions
from distill_factory.teachers.registry import get_teacher, validate_teacher_capabilities
from distill_factory.utils.logging import log_stage_metrics


def _dry_run_topk_output(record: dict[str, Any]) -> dict[str, Any]:
    k = int(record.get("top_k", 5))
    return {
        "top_k_ids": list(range(k)),
        "top_k_logprobs": [-(i + 1) * 0.1 for i in range(k)],
        "entropy": 0.0,
    }



def _slice_window_fields(window_record: dict[str, Any], start: int, end: int) -> None:
    """Slice position-aligned fields to an inclusive position window."""
    position_fields = (
        "top_k_ids",
        "top_k_logprobs",
        "per_token_entropy",
        "per_token_top1_gap",
        "per_token_token_ids",
        "per_token_valid_mask",
    )
    for field in position_fields:
        values = window_record.get(field)
        if isinstance(values, list):
            window_record[field] = values[start : end + 1]



def _build_selection_mask(record: dict[str, Any]) -> tuple[list[bool], dict[str, Any]]:
    per_token_entropy = record.get("per_token_entropy")
    per_token_top1_gap = record.get("per_token_top1_gap")
    entropy_threshold_raw = record.get("entropy_threshold")
    top1_gap_threshold_raw = record.get("top1_gap_threshold")
    entropy_threshold = None if entropy_threshold_raw is None else float(entropy_threshold_raw)
    top1_gap_threshold = None if top1_gap_threshold_raw is None else float(top1_gap_threshold_raw)

    if entropy_threshold is None and top1_gap_threshold is None:
        raise ValueError(
            "Stage A selection requires entropy_threshold and/or top1_gap_threshold when position filtering is enabled"
        )

    selection_combine_mode = str(record.get("selection_combine_mode", "union"))
    minimum_selected_positions = record.get("minimum_selected_positions_per_record")
    minimum_selected = None if minimum_selected_positions is None else int(minimum_selected_positions)

    mask = select_positions(
        per_token_entropy=per_token_entropy if isinstance(per_token_entropy, list) else None,
        per_token_top1_gap=per_token_top1_gap if isinstance(per_token_top1_gap, list) else None,
        entropy_threshold=entropy_threshold,
        top1_gap_threshold=top1_gap_threshold,
        combine_mode=selection_combine_mode,
        minimum_selected_positions=minimum_selected,
    )

    policy = {
        "combine_mode": selection_combine_mode,
        "entropy_threshold": entropy_threshold,
        "top1_gap_threshold": top1_gap_threshold,
        "selection_window_radius": int(record.get("selection_window_radius", 0)),
        "minimum_selected_positions_per_record": minimum_selected,
        "total_selected_positions_in_record": int(sum(1 for v in mask if v)),
    }
    return mask, policy





def _selection_requirements(records: list[dict[str, Any]]) -> tuple[bool, bool]:
    """Return whether per-token entropy/gap capabilities are required."""
    need_entropy = False
    need_gap = False
    for record in records:
        selection_mode = str(record.get("selection_mode", "none"))
        if selection_mode == "none":
            continue
        if record.get("entropy_threshold") is not None:
            need_entropy = True
        if record.get("top1_gap_threshold") is not None:
            need_gap = True
    return need_entropy, need_gap

def _apply_position_aware_export(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply optional Stage A position-aware export policies."""
    out: list[dict[str, Any]] = []

    for record in records:
        enable_position_filtering = bool(record.get("enable_position_filtering", False))
        selection_mode = str(record.get("selection_mode", "none"))

        if not enable_position_filtering or selection_mode == "none":
            out.append(record)
            continue

        if selection_mode == "position_mask":
            mask, policy = _build_selection_mask(record)
            selected_positions = [i for i, keep in enumerate(mask) if keep]
            full_record = dict(record)
            extra_metadata = dict(full_record.get("extra_metadata") or {})
            extra_metadata["selected_position_mask"] = [bool(v) for v in mask]
            extra_metadata["selected_positions"] = [int(i) for i in selected_positions]
            extra_metadata["selected_position_count"] = int(len(selected_positions))
            extra_metadata["selection_policy"] = {
                "selection_mode": selection_mode,
                **policy,
            }
            full_record["extra_metadata"] = extra_metadata
            out.append(full_record)
            continue

        if selection_mode == "selected_windows":
            mask, policy = _build_selection_mask(record)
            selection_window_radius = int(record.get("selection_window_radius", 0))
            windows = mask_to_windows(mask, radius=selection_window_radius)
            if not windows:
                continue

            for start, end in windows:
                window_record = dict(record)
                _slice_window_fields(window_record, start, end)

                selected_in_window = sum(1 for idx in range(start, end + 1) if mask[idx])
                extra_metadata = dict(window_record.get("extra_metadata") or {})
                extra_metadata["selected_window_start"] = int(start)
                extra_metadata["selected_window_end"] = int(end)
                extra_metadata["selected_position_count"] = int(selected_in_window)
                extra_metadata["selection_policy"] = {
                    "selection_mode": selection_mode,
                    **policy,
                }
                window_record["extra_metadata"] = extra_metadata
                out.append(window_record)
            continue

        # Unknown mode: preserve current dense behavior conservatively.
        out.append(record)

    return out



def run_stage_a(
    records: list[dict[str, Any]],
    teacher_name: str,
    mode: str,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Run stage A on chunk-local records."""
    if not records:
        return records

    if dry_run:
        outputs = [_dry_run_topk_output(r) for r in records]
    else:
        teacher = get_teacher(teacher_name)
        requires_hidden_summary = any(bool(r.get("extract_hidden_summary", False)) for r in records)
        requires_tokenizer_diagnostics = os.environ.get("DISTILL_FACTORY_LOG_TOKEN_LENGTHS", "0") == "1"

        if requires_hidden_summary and not bool(getattr(teacher, "supports_hidden_summary", lambda: False)()):
            raise ValueError(
                f"Stage A requested hidden_summary extraction, but teacher '{teacher_name}' "
                "does not support hidden summaries."
            )

        require_per_token_entropy, require_per_token_top1_gap = _selection_requirements(records)

        validate_teacher_capabilities(
            teacher,
            teacher_name,
            stage_name="stage_a",
            mode=mode,
            require_topk=True,
            require_hidden_summary=requires_hidden_summary,
            require_per_token_entropy=require_per_token_entropy,
            require_per_token_top1_gap=require_per_token_top1_gap,
        )
        teacher.prepare()
        try:
            if requires_tokenizer_diagnostics and not bool(getattr(teacher, "supports_tokenizer_diagnostics", lambda: False)()):
                raise ValueError(
                    f"Stage A token-length diagnostics requested (DISTILL_FACTORY_LOG_TOKEN_LENGTHS=1), "
                    f"but teacher '{teacher_name}' does not support tokenizer diagnostics."
                )
            outputs = teacher.infer_topk(records)
        finally:
            teacher.close()

    for record, output in zip(records, outputs):
        record["teacher_name"] = teacher_name
        record["stage_name"] = "stage_a"
        record["mode"] = mode
        record["top_k_ids"] = output.get("top_k_ids")
        record["top_k_logprobs"] = output.get("top_k_logprobs")
        record["entropy"] = output.get("entropy")
        record["per_token_entropy"] = output.get("per_token_entropy")
        record["per_token_top1_gap"] = output.get("per_token_top1_gap")
        record["per_token_token_ids"] = output.get("per_token_token_ids")
        record["per_token_valid_mask"] = output.get("per_token_valid_mask")
        meta = dict(record.get("extra_metadata") or {})
        if dry_run:
            meta["dry_run"] = True
            meta["dry_run_note"] = "teacher inference skipped"
        record["extra_metadata"] = meta

    summary = log_stage_metrics(records, stage_name="stage_a")
    for record in records:
        meta = dict(record.get("extra_metadata") or {})
        meta["teacher_sanity"] = summary
        record["extra_metadata"] = meta

    return _apply_position_aware_export(records)
