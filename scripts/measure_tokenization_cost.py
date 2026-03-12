#!/usr/bin/env python3
"""Measure tokenization cost for a chosen teacher backend on a small corpus subset."""

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
from distill_factory.teachers.hf_causal_lm import HFCausalLMTeacher
from distill_factory.teachers.llamacpp_server import LlamaCppServerTeacher
from distill_factory.teachers.vllm_causal_lm import VLLMCausalLMTeacher


def _sample_chunks(config_path: str, limit: int, input_path_override: str | None = None, file_glob_override: str | None = None) -> tuple[list[dict], object]:
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


def _build_teacher(cfg: object, backend: str):
    if backend == "hf":
        return HFCausalLMTeacher(
            model_name_or_path=cfg.stage_a.model_name_or_path,
            device_map=cfg.stage_a.device_map,
            hf_offload_layers=cfg.stage_a.hf_offload_layers,
            torch_dtype=cfg.stage_a.torch_dtype,
            max_context=cfg.stage_a.max_context,
            batch_size=cfg.stage_a.batch_size,
        ), int(cfg.stage_a.max_context), str(cfg.stage_a.model_name_or_path)

    if backend == "vllm":
        return VLLMCausalLMTeacher(
            model_name_or_path=cfg.stage_a.model_name_or_path,
            tensor_parallel_size=cfg.stage_a.tensor_parallel_size,
            dtype=cfg.stage_a.dtype,
            max_context=cfg.stage_a.max_context,
            batch_size=cfg.stage_a.batch_size,
            gpu_memory_utilization=cfg.stage_a.gpu_memory_utilization,
            trust_remote_code=cfg.stage_a.trust_remote_code,
        ), int(cfg.stage_a.max_context), str(cfg.stage_a.model_name_or_path)

    if backend == "llamacpp_server":
        model_label = cfg.stage_a.llama_model_hint if cfg.stage_a.llama_model_hint else cfg.stage_a.llama_base_url
        return LlamaCppServerTeacher(
            base_url=cfg.stage_a.llama_base_url,
            model_hint=cfg.stage_a.llama_model_hint,
            request_timeout=cfg.stage_a.llama_request_timeout,
            max_context=cfg.stage_a.max_context,
            default_top_k=cfg.stage_a.top_k,
            default_temperature=cfg.stage_a.temperature,
        ), int(cfg.stage_a.max_context), str(model_label)

    raise ValueError("--backend must be 'hf', 'vllm', or 'llamacpp_server'")


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure tokenizer cost on a small sampled subset.")
    parser.add_argument("--config", default="configs/examples/default.toml", help="Path to pipeline config.")
    parser.add_argument("--backend", choices=["hf", "vllm", "llamacpp_server"], default="hf", help="Tokenizer backend to use.")
    parser.add_argument("--sample-records", type=int, default=32, help="Number of chunk records to sample.")
    parser.add_argument("--input-path", default=None, help="Optional override for [data].input_path.")
    parser.add_argument("--file-glob", default=None, help="Optional override for [data].file_glob.")
    parser.add_argument("--near-limit-threshold", type=float, default=0.9, help="Fraction of max_context considered near-limit.")
    args = parser.parse_args()

    chunks, cfg = _sample_chunks(args.config, args.sample_records, args.input_path, args.file_glob)
    if not chunks:
        raise ValueError("No chunks available from current config input corpus")

    teacher, max_context, model_name = _build_teacher(cfg, args.backend)
    texts = [bytes(r["raw_bytes"]).decode("utf-8", errors="replace") for r in chunks]
    byte_lengths = [len(t.encode("utf-8", errors="replace")) for t in texts]

    try:
        token_lengths = teacher.token_lengths(texts)
    except NotImplementedError as exc:
        raise RuntimeError(
            f"Tokenization diagnostics are unavailable for backend '{args.backend}' with current runtime settings: {exc}"
        ) from exc
    finally:
        teacher.close()

    avg_bytes = statistics.mean(byte_lengths)
    avg_tokens = statistics.mean(token_lengths) if token_lengths else 0.0
    min_tokens = min(token_lengths) if token_lengths else 0
    max_tokens = max(token_lengths) if token_lengths else 0
    ratio = (avg_bytes / avg_tokens) if avg_tokens > 0 else 0.0

    near_cutoff = max(1, int(max_context * float(args.near_limit_threshold)))
    near_count = sum(1 for t in token_lengths if t >= near_cutoff)
    near_prop = near_count / len(token_lengths) if token_lengths else 0.0

    print("Tokenization cost summary")
    print(f"  backend: {args.backend}")
    print(f"  model_name_or_path: {model_name}")
    print(f"  sampled_records: {len(chunks)}")
    print(f"  max_context: {max_context}")
    print(f"  average_byte_length: {avg_bytes:.2f}")
    print(f"  average_token_length: {avg_tokens:.2f}")
    print(f"  min_token_length: {min_tokens}")
    print(f"  max_token_length: {max_tokens}")
    print(f"  byte_to_token_ratio: {ratio:.4f}")
    print(
        f"  near_max_context_proportion: {near_prop:.4f} "
        f"(threshold >= {near_cutoff} tokens, {args.near_limit_threshold:.2f} * max_context)"
    )


if __name__ == "__main__":
    main()
