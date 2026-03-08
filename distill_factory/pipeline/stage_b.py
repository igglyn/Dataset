"""Stage B execution (long-context structure teacher)."""

from __future__ import annotations

import os
from typing import Any

from distill_factory.data.chunking import build_long_context_records
from distill_factory.teachers.long_context import prepare_long_context_teacher_input
from distill_factory.teachers.registry import get_teacher, validate_teacher_capabilities
from distill_factory.utils.logging import log_stage_metrics


def _dry_run_topk_output(record: dict[str, Any]) -> dict[str, Any]:
    k = int(record.get("top_k", 5))
    return {
        "top_k_ids": list(range(k)),
        "top_k_logprobs": [-(i + 1) * 0.1 for i in range(k)],
        "entropy": 0.0,
    }


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


def _assert_stage_b_selection_export_not_requested(records: list[dict[str, Any]]) -> None:
    """Fail fast for Stage B selection export requests.

    Stage B currently supports long-context supervision and can require per-token
    teacher signals for capability validation, but it does not implement Stage-A-like
    selection-aware export modes (position_mask/selected_windows).
    """
    for record in records:
        selection_mode = str(record.get("selection_mode", "none"))
        enable_position_filtering = bool(record.get("enable_position_filtering", False))
        if enable_position_filtering or selection_mode in {"position_mask", "selected_windows"}:
            raise NotImplementedError(
                "Stage B selection-aware export is not implemented yet: "
                "requested selection_mode='"
                + selection_mode
                + "' with enable_position_filtering="
                + str(enable_position_filtering)
                + ". "
                "Use Stage A for selection export, or set [stage_b].selection_mode='none' "
                "and enable_position_filtering=false for dense Stage B output."
            )


def _validate_stage_b_window_metadata(
    *,
    record: dict[str, Any],
    view: dict[str, Any],
    prepared: dict[str, Any],
) -> None:
    target_start = int(record["byte_start"])
    target_end = int(record["byte_end"])
    window_start = int(view["window_byte_start"])
    window_end = int(view["window_byte_end"])
    prepared_start = int(prepared["target_start_offset"])
    prepared_end = int(prepared["target_end_offset"])
    teacher_bytes = bytes(prepared["teacher_input_bytes"])

    if not (window_start <= target_start <= target_end <= window_end):
        raise RuntimeError(
            "Stage B metadata violation: target span must remain inside the selected long-context window"
        )

    if not (0 <= prepared_start <= prepared_end <= len(teacher_bytes)):
        raise RuntimeError(
            "Stage B metadata violation: target offsets within teacher window are out of bounds"
        )

    trunc = prepared.get("truncation_metadata")
    if isinstance(trunc, dict):
        if int(trunc.get("final_window_bytes", len(teacher_bytes))) != len(teacher_bytes):
            raise RuntimeError(
                "Stage B metadata violation: truncation metadata final_window_bytes does not match teacher bytes"
            )


def run_stage_b(
    records: list[dict[str, Any]],
    teacher_name: str,
    mode: str,
    context_window: int,
    stride: int,
    max_teacher_context: int | None = None,
    window_policy: str = "center_target",
    target_region_policy: str = "preserve_full",
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Run stage B with document-aware long-context windows around each target chunk."""
    if not records:
        return records

    _assert_stage_b_selection_export_not_requested(records)

    long_views = build_long_context_records(
        records=records,
        context_window=context_window,
        stride=stride,
    )

    infer_inputs: list[dict[str, Any]] = []
    prepared_contexts: list[dict[str, Any]] = []
    teacher_context = context_window if max_teacher_context is None else int(max_teacher_context)

    for record, view in zip(records, long_views):
        # Interpretation note:
        # `prepare_long_context_teacher_input(..., target_region_policy="preserve_full")`
        # may legally return a window larger than `max_teacher_context` when the target
        # span itself exceeds that limit. We keep this behavior and validate offsets.
        prepared = prepare_long_context_teacher_input(
            window_raw_bytes=view["window_raw_bytes"],
            target_start_offset=int(view["target_byte_start_in_window"]),
            target_end_offset=int(view["target_byte_end_in_window"]),
            max_teacher_context=teacher_context,
            window_policy=window_policy,
            target_region_policy=target_region_policy,
        )
        _validate_stage_b_window_metadata(record=record, view=view, prepared=prepared)
        prepared_contexts.append(prepared)

        infer_inputs.append(
            {
                "doc_id": record["doc_id"],
                "chunk_index": record["chunk_index"],
                "raw_bytes": prepared["teacher_input_bytes"],
                "teacher_input_bytes": prepared["teacher_input_bytes"],
                "teacher_input_text": prepared["teacher_input_text"],
                "target_start_offset": prepared["target_start_offset"],
                "target_end_offset": prepared["target_end_offset"],
                "long_context_truncation": prepared["truncation_metadata"],
                "top_k": record.get("top_k", 5),
            }
        )

    if dry_run:
        outputs = [_dry_run_topk_output(r) for r in infer_inputs]
    else:
        teacher = get_teacher(teacher_name)
        requires_hidden_summary = any(bool(r.get("extract_hidden_summary", False)) for r in records)
        requires_tokenizer_diagnostics = os.environ.get("DISTILL_FACTORY_LOG_TOKEN_LENGTHS", "0") == "1"

        if requires_hidden_summary and not bool(getattr(teacher, "supports_hidden_summary", lambda: False)()):
            raise ValueError(
                f"Stage B requested hidden_summary extraction, but teacher '{teacher_name}' "
                "does not support hidden summaries."
            )

        require_per_token_entropy, require_per_token_top1_gap = _selection_requirements(records)

        validate_teacher_capabilities(
            teacher,
            teacher_name,
            stage_name="stage_b",
            mode=mode,
            require_topk=True,
            require_long_context=True,
            require_hidden_summary=requires_hidden_summary,
            require_per_token_entropy=require_per_token_entropy,
            require_per_token_top1_gap=require_per_token_top1_gap,
        )
        teacher.prepare()
        try:
            if requires_tokenizer_diagnostics and not bool(getattr(teacher, "supports_tokenizer_diagnostics", lambda: False)()):
                raise ValueError(
                    f"Stage B token-length diagnostics requested (DISTILL_FACTORY_LOG_TOKEN_LENGTHS=1), "
                    f"but teacher '{teacher_name}' does not support tokenizer diagnostics."
                )
            outputs = teacher.infer_topk(infer_inputs)
        finally:
            teacher.close()

    for record, view, prepared, output in zip(records, long_views, prepared_contexts, outputs):
        meta = dict(record.get("extra_metadata") or {})
        meta["stage_b_context"] = {
            "context_window": context_window,
            "stride": stride,
            "window_byte_start": view["window_byte_start"],
            "window_byte_end": view["window_byte_end"],
            "target_byte_start_in_window": prepared["target_start_offset"],
            "target_byte_end_in_window": prepared["target_end_offset"],
            "max_teacher_context": teacher_context,
            "window_policy": window_policy,
            "target_region_policy": target_region_policy,
            "long_context_text": prepared["teacher_input_text"],
            "truncation": prepared["truncation_metadata"],
        }

        record["target_doc_id"] = str(record.get("doc_id"))
        record["target_chunk_index"] = int(record.get("chunk_index"))
        record["target_byte_start"] = int(record.get("byte_start"))
        record["target_byte_end"] = int(record.get("byte_end"))
        record["teacher_window_byte_start"] = int(view["window_byte_start"])
        record["teacher_window_byte_end"] = int(view["window_byte_end"])
        record["target_start_offset_within_window"] = int(prepared["target_start_offset"])
        record["target_end_offset_within_window"] = int(prepared["target_end_offset"])

        record["teacher_name"] = teacher_name
        record["stage_name"] = "stage_b"
        record["mode"] = mode
        record["top_k_ids"] = output.get("top_k_ids")
        record["top_k_logprobs"] = output.get("top_k_logprobs")
        record["entropy"] = output.get("entropy")
        if dry_run:
            meta["dry_run"] = True
            meta["dry_run_note"] = "teacher inference skipped"
        record["extra_metadata"] = meta

    summary = log_stage_metrics(records, stage_name="stage_b")
    for record in records:
        meta = dict(record.get("extra_metadata") or {})
        meta["teacher_sanity"] = summary
        record["extra_metadata"] = meta

    return records
