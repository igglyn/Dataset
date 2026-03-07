"""Stage C execution (refinement teacher)."""

from __future__ import annotations

import os
from typing import Any

from distill_factory.pipeline.prompts import build_prompt_record
from distill_factory.teachers.registry import get_teacher, validate_teacher_capabilities


def _build_structured_prompt(
    record: dict[str, Any],
    template_name: str,
    template_kwargs: dict[str, Any] | None,
    deterministic: bool,
) -> dict[str, Any]:
    return build_prompt_record(
        record=record,
        template_name=template_name,
        template_kwargs=template_kwargs,
        deterministic=deterministic,
    )




def _validate_structured_output_record(
    *,
    structured_output: dict[str, Any],
    template_name: str,
    deterministic: bool,
) -> None:
    """Fail fast when Stage C structured outputs are missing inspectable fields."""
    required = ("task_type", "prompt_text", "completion_text", "teacher_metadata")
    for key in required:
        if key not in structured_output:
            raise RuntimeError(f"Stage C structured output missing required field: {key}")

    if not isinstance(structured_output.get("task_type"), str) or not structured_output["task_type"].strip():
        raise RuntimeError("Stage C structured output task_type must be a non-empty string")
    if not isinstance(structured_output.get("prompt_text"), str) or not structured_output["prompt_text"].strip():
        raise RuntimeError("Stage C structured output prompt_text must be a non-empty string")
    if not isinstance(structured_output.get("completion_text"), str):
        raise RuntimeError("Stage C structured output completion_text must be a string")

def _dry_run_topk_output(record: dict[str, Any]) -> dict[str, Any]:
    k = int(record.get("top_k", 5))
    return {
        "top_k_ids": list(range(k)),
        "top_k_logprobs": [-(i + 1) * 0.1 for i in range(k)],
        "entropy": 0.0,
    }


def run_stage_c(
    records: list[dict[str, Any]],
    teacher_name: str,
    mode: str,
    template_name: str = "summarize_chunk",
    template_kwargs: dict[str, Any] | None = None,
    deterministic: bool = True,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Run stage C refinement, attaching structured outputs when requested."""
    if not records:
        return records

    structured_inputs: list[dict[str, Any]] = []
    if mode == "structured_outputs":
        structured_inputs = [
            _build_structured_prompt(
                r,
                template_name=template_name,
                template_kwargs=template_kwargs,
                deterministic=deterministic,
            )
            for r in records
        ]

    if dry_run:
        if mode == "structured_outputs":
            outputs = [
                {
                    "task_type": s.get("task_type"),
                    "prompt_text": s.get("prompt_text"),
                    "completion_text": "",
                    "teacher_metadata": {"dry_run": True, "note": "teacher inference skipped"},
                }
                for s in structured_inputs
            ]
        else:
            outputs = [_dry_run_topk_output(r) for r in records]
    else:
        teacher = get_teacher(teacher_name)
        requires_hidden_summary = any(bool(r.get("extract_hidden_summary", False)) for r in records)
        requires_tokenizer_diagnostics = os.environ.get("DISTILL_FACTORY_LOG_TOKEN_LENGTHS", "0") == "1"

        if mode == "structured_outputs" and not bool(getattr(teacher, "supports_structured", lambda: False)()):
            raise ValueError(
                f"Stage C mode='structured_outputs' requested, but teacher '{teacher_name}' "
                "does not support structured generation."
            )
        if requires_hidden_summary and not bool(getattr(teacher, "supports_hidden_summary", lambda: False)()):
            raise ValueError(
                f"Stage C requested hidden_summary extraction, but teacher '{teacher_name}' "
                "does not support hidden summaries."
            )

        if mode == "structured_outputs":
            validate_teacher_capabilities(
                teacher,
                teacher_name,
                stage_name="stage_c",
                mode=mode,
                require_structured=True,
            )
        else:
            validate_teacher_capabilities(
                teacher,
                teacher_name,
                stage_name="stage_c",
                mode=mode,
                require_topk=True,
                require_hidden_summary=requires_hidden_summary,
            )

        teacher.prepare()
        try:
            if requires_tokenizer_diagnostics and not bool(getattr(teacher, "supports_tokenizer_diagnostics", lambda: False)()):
                raise ValueError(
                    f"Stage C token-length diagnostics requested (DISTILL_FACTORY_LOG_TOKEN_LENGTHS=1), "
                    f"but teacher '{teacher_name}' does not support tokenizer diagnostics."
                )
            if mode == "structured_outputs":
                outputs = teacher.infer_structured(structured_inputs)
            else:
                outputs = teacher.infer_topk(records)
        finally:
            teacher.close()

    for idx, (record, output) in enumerate(zip(records, outputs)):
        meta = dict(record.get("extra_metadata") or {})
        record["teacher_name"] = teacher_name
        record["stage_name"] = "stage_c"
        record["mode"] = mode

        if mode == "structured_outputs":
            prompt_payload = structured_inputs[idx]

            structured_output = {
                "task_type": str(output.get("task_type", prompt_payload["task_type"])),
                "prompt_text": str(output.get("prompt_text", prompt_payload["prompt_text"])),
                "completion_text": str(output.get("completion_text", "")),
                "teacher_metadata": output.get("teacher_metadata")
                if isinstance(output.get("teacher_metadata"), dict) or output.get("teacher_metadata") is None
                else None,
            }
            _validate_structured_output_record(
                structured_output=structured_output,
                template_name=template_name,
                deterministic=deterministic,
            )
            record["structured_output"] = structured_output
            meta["structured_output"] = structured_output
            meta["stage_c_template"] = {
                "template_name": template_name,
                "template_kwargs": dict(template_kwargs or {}),
                "deterministic": bool(deterministic),
                "template_metadata": prompt_payload.get("template_metadata", {}),
                "record_index": idx,
            }
            record["top_k_ids"] = None
            record["top_k_logprobs"] = None
            record["entropy"] = None
        else:
            record["structured_output"] = None
            record["top_k_ids"] = output.get("top_k_ids")
            record["top_k_logprobs"] = output.get("top_k_logprobs")
            record["entropy"] = output.get("entropy")

        if dry_run:
            meta["dry_run"] = True
            meta["dry_run_note"] = "teacher inference skipped"
        record["extra_metadata"] = meta
    return records
