#!/usr/bin/env python3
"""Lightweight sanity comparison between two backend configurations.

This is a parity *sanity* tool for tiny samples, not a benchmark and not a
numerical-equivalence test.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import statistics
import sys
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from distill_factory.config.schema import load_config
from distill_factory.data.chunking import chunk_documents
from distill_factory.data.ingest import ingest_documents
from distill_factory.pipeline.stage_a import selection_artifacts_for_record
from distill_factory.teachers.hf_causal_lm import HFCausalLMTeacher
from distill_factory.teachers.llamacpp_server import LlamaCppServerTeacher
from distill_factory.teachers.vllm_causal_lm import VLLMCausalLMTeacher


@dataclass(slots=True)
class RunSummary:
    label: str
    backend: str
    model_label: str
    record_count: int
    topk_present_count: int
    token_length_count: int
    token_length_min: int | None
    token_length_max: int | None
    token_length_avg: float | None
    entropy_count: int
    entropy_min: float | None
    entropy_max: float | None
    entropy_avg: float | None
    per_token_entropy_present_count: int
    per_token_top1_gap_present_count: int
    per_token_entropy_plausible_len_count: int
    per_token_top1_gap_plausible_len_count: int
    selection_mode: str
    selection_enabled: bool
    selection_metadata_possible_count: int


def _sample_chunks(config_path: str, limit: int, input_path_override: str | None = None, file_glob_override: str | None = None) -> tuple[list[dict[str, Any]], Any]:
    cfg = load_config(config_path)
    input_path = input_path_override or cfg.data.input_path
    file_glob = file_glob_override or cfg.data.file_glob
    docs = ingest_documents(
        input_path=input_path,
        file_glob=file_glob,
        encoding=cfg.data.encoding,
        normalize_newlines=cfg.input.normalize_newlines,
    )
    chunks = chunk_documents(
        documents=docs,
        chunk_bytes=cfg.data.chunk_bytes,
        overlap_bytes=cfg.data.overlap_bytes,
        encoding=cfg.data.encoding,
    )
    return chunks[: max(1, limit)], cfg


def _selection_requirements_from_cfg(cfg: Any) -> tuple[bool, bool]:
    if not bool(cfg.stage_a.enable_position_filtering) or str(cfg.stage_a.selection_mode) == "none":
        return False, False
    return cfg.stage_a.entropy_threshold is not None, cfg.stage_a.top1_gap_threshold is not None


def _build_teacher(cfg: Any, backend: str):
    require_entropy, require_gap = _selection_requirements_from_cfg(cfg)
    if backend == "hf":
        return HFCausalLMTeacher(
            model_name_or_path=cfg.stage_a.model_name_or_path,
            device_map=cfg.stage_a.device_map,
            hf_offload_layers=cfg.stage_a.hf_offload_layers,
            torch_dtype=cfg.stage_a.torch_dtype,
            max_context=cfg.stage_a.max_context,
            batch_size=cfg.stage_a.batch_size,
            emit_per_token_entropy=require_entropy,
            emit_per_token_top1_gap=require_gap,
        ), str(cfg.stage_a.model_name_or_path)

    if backend == "vllm":
        return VLLMCausalLMTeacher(
            model_name_or_path=cfg.stage_a.model_name_or_path,
            tensor_parallel_size=cfg.stage_a.tensor_parallel_size,
            dtype=cfg.stage_a.dtype,
            max_context=cfg.stage_a.max_context,
            batch_size=cfg.stage_a.batch_size,
            gpu_memory_utilization=cfg.stage_a.gpu_memory_utilization,
            trust_remote_code=cfg.stage_a.trust_remote_code,
            emit_per_token_entropy=require_entropy,
            emit_per_token_top1_gap=require_gap,
        ), str(cfg.stage_a.model_name_or_path)

    if backend == "llamacpp_server":
        label = cfg.stage_a.llama_model_hint if cfg.stage_a.llama_model_hint else cfg.stage_a.llama_base_url
        return LlamaCppServerTeacher(
            base_url=cfg.stage_a.llama_base_url,
            model_hint=cfg.stage_a.llama_model_hint,
            request_timeout=cfg.stage_a.llama_request_timeout,
            max_context=cfg.stage_a.max_context,
            default_top_k=cfg.stage_a.top_k,
            default_temperature=cfg.stage_a.temperature,
            emit_per_token_entropy=require_entropy,
            emit_per_token_top1_gap=require_gap,
        ), str(label)

    raise ValueError(f"Unsupported backend: {backend}")


def _position_count(top_k_ids: Any) -> int:
    if not isinstance(top_k_ids, list):
        return 0
    return len(top_k_ids)


def _plausible_len(values: Any, positions: int) -> bool:
    return isinstance(values, list) and (positions == 0 or len(values) == positions)


def _selection_record_from_output(cfg: Any, out: dict[str, Any]) -> dict[str, Any]:
    return {
        "top_k_ids": out.get("top_k_ids"),
        "top_k_logprobs": out.get("top_k_logprobs"),
        "per_token_entropy": out.get("per_token_entropy"),
        "per_token_top1_gap": out.get("per_token_top1_gap"),
        "enable_position_filtering": bool(cfg.stage_a.enable_position_filtering),
        "selection_mode": str(cfg.stage_a.selection_mode),
        "entropy_threshold": cfg.stage_a.entropy_threshold,
        "top1_gap_threshold": cfg.stage_a.top1_gap_threshold,
        "selection_window_radius": int(cfg.stage_a.selection_window_radius),
        "minimum_selected_positions_per_record": cfg.stage_a.minimum_selected_positions_per_record,
    }


def _run_backend(label: str, cfg: Any, backend: str, chunks: list[dict[str, Any]]) -> RunSummary:
    teacher, model_label = _build_teacher(cfg, backend)
    records = [{"raw_bytes": bytes(c["raw_bytes"]), "top_k": int(cfg.stage_a.top_k)} for c in chunks]

    teacher.prepare()
    try:
        outputs = teacher.infer_topk(records)
    finally:
        teacher.close()

    topk_present_count = 0
    token_lengths: list[int] = []
    entropies: list[float] = []
    per_token_entropy_present_count = 0
    per_token_top1_gap_present_count = 0
    per_token_entropy_plausible_len_count = 0
    per_token_top1_gap_plausible_len_count = 0
    selection_metadata_possible_count = 0

    for out in outputs:
        top_k_ids = out.get("top_k_ids")
        if top_k_ids is not None and out.get("top_k_logprobs") is not None:
            topk_present_count += 1

        tlen = out.get("teacher_input_token_length")
        if isinstance(tlen, (int, float)):
            token_lengths.append(int(tlen))

        entropy = out.get("entropy")
        if isinstance(entropy, (int, float)):
            entropies.append(float(entropy))

        positions = _position_count(top_k_ids)
        per_entropy = out.get("per_token_entropy")
        per_gap = out.get("per_token_top1_gap")

        if per_entropy is not None:
            per_token_entropy_present_count += 1
            if _plausible_len(per_entropy, positions):
                per_token_entropy_plausible_len_count += 1

        if per_gap is not None:
            per_token_top1_gap_present_count += 1
            if _plausible_len(per_gap, positions):
                per_token_top1_gap_plausible_len_count += 1

        artifacts = selection_artifacts_for_record(_selection_record_from_output(cfg, out))
        if str(artifacts["selection_mode"]) == "position_mask":
            selection_metadata_possible_count += 1
        elif str(artifacts["selection_mode"]) == "selected_windows" and len(artifacts["windows"]) > 0:
            selection_metadata_possible_count += 1

    return RunSummary(
        label=label,
        backend=backend,
        model_label=model_label,
        record_count=len(outputs),
        topk_present_count=topk_present_count,
        token_length_count=len(token_lengths),
        token_length_min=min(token_lengths) if token_lengths else None,
        token_length_max=max(token_lengths) if token_lengths else None,
        token_length_avg=statistics.mean(token_lengths) if token_lengths else None,
        entropy_count=len(entropies),
        entropy_min=min(entropies) if entropies else None,
        entropy_max=max(entropies) if entropies else None,
        entropy_avg=statistics.mean(entropies) if entropies else None,
        per_token_entropy_present_count=per_token_entropy_present_count,
        per_token_top1_gap_present_count=per_token_top1_gap_present_count,
        per_token_entropy_plausible_len_count=per_token_entropy_plausible_len_count,
        per_token_top1_gap_plausible_len_count=per_token_top1_gap_plausible_len_count,
        selection_mode=str(cfg.stage_a.selection_mode),
        selection_enabled=bool(cfg.stage_a.enable_position_filtering) and str(cfg.stage_a.selection_mode) != "none",
        selection_metadata_possible_count=selection_metadata_possible_count,
    )


def _fmt_opt_float(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _print_summary(summary: RunSummary) -> None:
    print(f"[{summary.label}] backend={summary.backend} model={summary.model_label}")
    print(f"  records: {summary.record_count}")
    print(f"  top-k fields present: {summary.topk_present_count}/{summary.record_count}")
    print(
        "  token length stats: "
        f"count={summary.token_length_count} "
        f"min={summary.token_length_min if summary.token_length_min is not None else 'n/a'} "
        f"max={summary.token_length_max if summary.token_length_max is not None else 'n/a'} "
        f"avg={_fmt_opt_float(summary.token_length_avg, 2)}"
    )
    print(
        "  entropy stats: "
        f"count={summary.entropy_count} "
        f"min={_fmt_opt_float(summary.entropy_min)} "
        f"max={_fmt_opt_float(summary.entropy_max)} "
        f"avg={_fmt_opt_float(summary.entropy_avg)}"
    )
    print(
        "  per-token structural: "
        f"entropy_present={summary.per_token_entropy_present_count}/{summary.record_count} "
        f"gap_present={summary.per_token_top1_gap_present_count}/{summary.record_count} "
        f"entropy_len_plausible={summary.per_token_entropy_plausible_len_count}/{summary.per_token_entropy_present_count or 1} "
        f"gap_len_plausible={summary.per_token_top1_gap_plausible_len_count}/{summary.per_token_top1_gap_present_count or 1}"
    )
    if summary.selection_enabled:
        print(
            "  selection metadata feasibility: "
            f"{summary.selection_metadata_possible_count}/{summary.record_count} "
            f"(mode={summary.selection_mode})"
        )


def _status(level: str, msg: str) -> None:
    print(f"{level}: {msg}")


def _print_comparison(left: RunSummary, right: RunSummary) -> None:
    print("\n=== Parity sanity comparison ===")
    _status("PASS" if left.record_count == right.record_count else "FAIL", f"record_count left={left.record_count} right={right.record_count}")

    _status(
        "PASS" if left.topk_present_count == left.record_count and right.topk_present_count == right.record_count else "WARN",
        (
            "top-k field presence "
            f"left={left.topk_present_count}/{left.record_count} "
            f"right={right.topk_present_count}/{right.record_count}"
        ),
    )

    _status(
        "PASS" if left.per_token_entropy_present_count == right.per_token_entropy_present_count else "WARN",
        (
            "per_token_entropy presence "
            f"left={left.per_token_entropy_present_count}/{left.record_count} "
            f"right={right.per_token_entropy_present_count}/{right.record_count}"
        ),
    )
    _status(
        "PASS" if left.per_token_top1_gap_present_count == right.per_token_top1_gap_present_count else "WARN",
        (
            "per_token_top1_gap presence "
            f"left={left.per_token_top1_gap_present_count}/{left.record_count} "
            f"right={right.per_token_top1_gap_present_count}/{right.record_count}"
        ),
    )

    left_len_ok = left.per_token_entropy_present_count == 0 or left.per_token_entropy_plausible_len_count == left.per_token_entropy_present_count
    right_len_ok = right.per_token_entropy_present_count == 0 or right.per_token_entropy_plausible_len_count == right.per_token_entropy_present_count
    _status(
        "PASS" if left_len_ok and right_len_ok else "FAIL",
        (
            "per_token_entropy length plausibility vs top-k positions "
            f"left={left.per_token_entropy_plausible_len_count}/{left.per_token_entropy_present_count} "
            f"right={right.per_token_entropy_plausible_len_count}/{right.per_token_entropy_present_count}"
        ),
    )

    left_gap_len_ok = left.per_token_top1_gap_present_count == 0 or left.per_token_top1_gap_plausible_len_count == left.per_token_top1_gap_present_count
    right_gap_len_ok = right.per_token_top1_gap_present_count == 0 or right.per_token_top1_gap_plausible_len_count == right.per_token_top1_gap_present_count
    _status(
        "PASS" if left_gap_len_ok and right_gap_len_ok else "FAIL",
        (
            "per_token_top1_gap length plausibility vs top-k positions "
            f"left={left.per_token_top1_gap_plausible_len_count}/{left.per_token_top1_gap_present_count} "
            f"right={right.per_token_top1_gap_plausible_len_count}/{right.per_token_top1_gap_present_count}"
        ),
    )

    if left.selection_enabled or right.selection_enabled:
        _status(
            "PASS" if left.selection_enabled == right.selection_enabled else "WARN",
            f"selection enabled flags left={left.selection_enabled} right={right.selection_enabled}",
        )
        _status(
            "PASS" if left.selection_metadata_possible_count > 0 and right.selection_metadata_possible_count > 0 else "WARN",
            (
                "selection metadata feasibility when enabled "
                f"left={left.selection_metadata_possible_count}/{left.record_count} "
                f"right={right.selection_metadata_possible_count}/{right.record_count}"
            ),
        )

    print(
        "\nNote: this checks structural parity for selection-aware export readiness; "
        "it does not require exact logit/probability equality."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate backend parity on a tiny shared sample.")
    parser.add_argument("--left-config", required=True, help="Left config path")
    parser.add_argument("--right-config", required=True, help="Right config path")
    parser.add_argument("--left-backend", choices=["hf", "vllm", "llamacpp_server"], default=None)
    parser.add_argument("--right-backend", choices=["hf", "vllm", "llamacpp_server"], default=None)
    parser.add_argument("--sample-records", type=int, default=8, help="Tiny sample size for sanity checks")
    parser.add_argument("--input-path", default=None, help="Optional shared input-path override")
    parser.add_argument("--file-glob", default=None, help="Optional shared file-glob override")
    args = parser.parse_args()

    chunks, left_cfg = _sample_chunks(args.left_config, args.sample_records, args.input_path, args.file_glob)
    if not chunks:
        raise ValueError("No chunks available for parity validation sample")

    right_cfg = load_config(args.right_config)

    left_backend = args.left_backend or str(left_cfg.stage_a.backend_type)
    right_backend = args.right_backend or str(right_cfg.stage_a.backend_type)

    left_summary = _run_backend("left", left_cfg, left_backend, chunks)
    right_summary = _run_backend("right", right_cfg, right_backend, chunks)

    print("=== Backend parity sanity ===")
    print(f"sample_records={len(chunks)}")
    _print_summary(left_summary)
    _print_summary(right_summary)
    _print_comparison(left_summary, right_summary)


if __name__ == "__main__":
    main()
