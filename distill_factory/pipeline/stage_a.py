"""Stage A execution (bulk grounding teacher)."""

from __future__ import annotations

import os
from typing import Any

from distill_factory.teachers.registry import get_teacher, validate_teacher_capabilities
from distill_factory.utils.logging import log_stage_metrics


def _dry_run_topk_output(record: dict[str, Any]) -> dict[str, Any]:
    k = int(record.get("top_k", 5))
    return {
        "top_k_ids": list(range(k)),
        "top_k_logprobs": [-(i + 1) * 0.1 for i in range(k)],
        "entropy": 0.0,
    }


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

        validate_teacher_capabilities(
            teacher,
            teacher_name,
            stage_name="stage_a",
            mode=mode,
            require_topk=True,
            require_hidden_summary=requires_hidden_summary,
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

    return records
