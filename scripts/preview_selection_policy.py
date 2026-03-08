#!/usr/bin/env python3
"""Preview Stage A selection policy behavior on a tiny sample.

This is a diagnostic helper (not a benchmark): it runs a small Stage A teacher pass,
computes selection masks/windows, and prints practical compression/selection signals
without writing a full dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from distill_factory.config.schema import load_config
from distill_factory.data.chunking import chunk_documents
from distill_factory.data.ingest import ingest_documents
from distill_factory.data.selection import mask_to_windows, select_positions
from distill_factory.teachers.hf_causal_lm import HFCausalLMTeacher
from distill_factory.teachers.llamacpp_server import LlamaCppServerTeacher
from distill_factory.teachers.vllm_causal_lm import VLLMCausalLMTeacher


def _sample_chunks(config_path: str, limit: int, input_path_override: str | None = None, file_glob_override: str | None = None):
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


def _build_stage_a_teacher(cfg):
    backend = str(cfg.stage_a.backend_type)
    if backend == "hf":
        return HFCausalLMTeacher(
            model_name_or_path=cfg.stage_a.model_name_or_path,
            device_map=cfg.stage_a.device_map,
            torch_dtype=cfg.stage_a.torch_dtype,
            max_context=cfg.stage_a.max_context,
            batch_size=cfg.stage_a.batch_size,
            emit_per_token_entropy=True,
            emit_per_token_top1_gap=True,
        )

    if backend == "vllm":
        return VLLMCausalLMTeacher(
            model_name_or_path=cfg.stage_a.model_name_or_path,
            tensor_parallel_size=cfg.stage_a.tensor_parallel_size,
            dtype=cfg.stage_a.dtype,
            max_context=cfg.stage_a.max_context,
            batch_size=cfg.stage_a.batch_size,
            gpu_memory_utilization=cfg.stage_a.gpu_memory_utilization,
            trust_remote_code=cfg.stage_a.trust_remote_code,
            emit_per_token_entropy=True,
            emit_per_token_top1_gap=True,
        )

    if backend == "llamacpp_server":
        return LlamaCppServerTeacher(
            base_url=cfg.stage_a.llama_base_url,
            model_hint=cfg.stage_a.llama_model_hint,
            request_timeout=cfg.stage_a.llama_request_timeout,
            max_context=cfg.stage_a.max_context,
            default_top_k=cfg.stage_a.top_k,
            default_temperature=cfg.stage_a.temperature,
            emit_per_token_entropy=True,
            emit_per_token_top1_gap=True,
        )

    raise ValueError(f"Unsupported stage_a.backend_type: {backend}")


def _selection_driver(cfg) -> str:
    has_entropy = cfg.stage_a.entropy_threshold is not None
    has_gap = cfg.stage_a.top1_gap_threshold is not None
    if has_entropy and has_gap:
        return "both"
    if has_entropy:
        return "entropy"
    if has_gap:
        return "gap"
    return "none"


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview Stage A selection policy on a tiny sample.")
    parser.add_argument("--config", default="configs/examples/default.toml", help="Path to pipeline config.")
    parser.add_argument("--sample-records", type=int, default=4, help="Number of chunk records to sample.")
    parser.add_argument("--input-path", default=None, help="Optional override for [data].input_path.")
    parser.add_argument("--file-glob", default=None, help="Optional override for [data].file_glob.")
    args = parser.parse_args()

    chunks, cfg = _sample_chunks(args.config, args.sample_records, args.input_path, args.file_glob)
    if not chunks:
        raise ValueError("No chunks available from current config input corpus")

    teacher = _build_stage_a_teacher(cfg)
    records = []
    for chunk in chunks:
        records.append(
            {
                "raw_bytes": bytes(chunk["raw_bytes"]),
                "top_k": int(cfg.stage_a.top_k),
                "emit_per_token_entropy": True,
                "emit_per_token_top1_gap": True,
            }
        )

    try:
        teacher.prepare()
        outputs = teacher.infer_topk(records)
    finally:
        teacher.close()

    selection_mode = str(cfg.stage_a.selection_mode)
    entropy_threshold = cfg.stage_a.entropy_threshold
    top1_gap_threshold = cfg.stage_a.top1_gap_threshold
    radius = int(cfg.stage_a.selection_window_radius)
    minimum_selected = cfg.stage_a.minimum_selected_positions_per_record

    print("Selection policy preview (Stage A)")
    print(f"  config: {args.config}")
    print(f"  sampled_records: {len(outputs)}")
    print(f"  backend_type: {cfg.stage_a.backend_type}")
    print(f"  selection_mode: {selection_mode}")
    print(f"  driven_by: {_selection_driver(cfg)}")
    print(f"  entropy_threshold: {entropy_threshold}")
    print(f"  top1_gap_threshold: {top1_gap_threshold}")
    print(f"  selection_window_radius: {radius}")
    print(f"  minimum_selected_positions_per_record: {minimum_selected}")

    compression_ratios: list[float] = []
    selected_counts: list[int] = []

    for idx, out in enumerate(outputs):
        per_entropy = out.get("per_token_entropy")
        per_gap = out.get("per_token_top1_gap")
        top_k_ids = out.get("top_k_ids")

        dense_len = len(top_k_ids) if isinstance(top_k_ids, list) else 0
        mask = select_positions(
            per_token_entropy=per_entropy if isinstance(per_entropy, list) else None,
            per_token_top1_gap=per_gap if isinstance(per_gap, list) else None,
            entropy_threshold=entropy_threshold,
            top1_gap_threshold=top1_gap_threshold,
            combine_mode="union",
            minimum_selected_positions=minimum_selected,
        )
        windows = mask_to_windows(mask, radius=radius)

        selected_count = sum(1 for v in mask if v)
        window_tokens = sum((end - start + 1) for start, end in windows)
        ratio = (float(window_tokens) / float(dense_len)) if dense_len > 0 else 0.0

        selected_counts.append(selected_count)
        compression_ratios.append(ratio)

        print(f"  record[{idx}] dense_positions={dense_len} selected_positions={selected_count} windows={windows}")

    avg_selected = statistics.mean(selected_counts) if selected_counts else 0.0
    avg_ratio = statistics.mean(compression_ratios) if compression_ratios else 0.0
    print(f"  average_selected_position_count: {avg_selected:.2f}")
    print(f"  average_compression_ratio_vs_dense_chunk: {avg_ratio:.4f}")


if __name__ == "__main__":
    main()
