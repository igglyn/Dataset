#!/usr/bin/env python3
"""Run a small reproducible benchmark to measure where pipeline time is spent."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from distill_factory.config.schema import load_config
from distill_factory.pipeline.orchestrator import run_pipeline
from distill_factory.storage.reader import read_jsonl_records
from distill_factory.utils.logging import format_timing_report


def _toml_literal(value: Any) -> str:
    if value is None:
        return "\"\""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, list):
        return "[" + ", ".join(_toml_literal(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{k} = {_toml_literal(v)}" for k, v in value.items()) + " }"
    raise TypeError(f"Unsupported TOML type for {value!r}")


def _write_toml(path: Path, cfg: dict[str, Any]) -> None:
    lines: list[str] = []
    for section in ("data", "input", "output", "stage_a", "stage_b", "stage_c"):
        lines.append(f"[{section}]")
        for k, v in cfg.get(section, {}).items():
            if v is None:
                continue
            lines.append(f"{k} = {_toml_literal(v)}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _copy_subset(input_path: Path, file_glob: str, limit: int, dest: Path) -> tuple[Path, int]:
    files = sorted([p for p in input_path.glob(file_glob) if p.is_file()], key=lambda p: p.as_posix())
    selected = files[:limit]
    if not selected:
        raise ValueError(f"No input files matched {file_glob!r} under {input_path}")

    subset_dir = dest / "benchmark_input"
    subset_dir.mkdir(parents=True, exist_ok=True)
    for src in selected:
        out = subset_dir / src.name
        if out.exists():
            out = subset_dir / f"{src.stem}_{len(list(subset_dir.glob(src.stem + '*')))}{src.suffix}"
        shutil.copy2(src, out)
    return subset_dir, len(selected)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark small pipeline runs and report stage timings.")
    parser.add_argument("--config", default="configs/examples/default.toml", help="Base config path.")
    parser.add_argument("--records", type=int, default=16, help="Number of input docs to include in benchmark subset.")
    parser.add_argument("--output-root", default="data/benchmark", help="Directory root for benchmark artifacts.")
    parser.add_argument("--input-path", default=None, help="Optional override for [data].input_path.")
    parser.add_argument("--file-glob", default=None, help="Optional override for [data].file_glob.")
    parser.add_argument("--dry-run", action="store_true", help="Use dry-run mode (skips teacher inference).")
    args = parser.parse_args()

    if args.records <= 0:
        raise ValueError("--records must be > 0")

    cfg = load_config(args.config)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.output_root) / f"pipeline_benchmark_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input_path) if args.input_path else Path(cfg.data.input_path)
    file_glob = args.file_glob if args.file_glob else cfg.data.file_glob
    subset_input, subset_count = _copy_subset(input_path, file_glob, args.records, run_dir)

    bench_cfg: dict[str, Any] = {
        "data": {
            "input_path": str(subset_input),
            "file_glob": file_glob,
            "encoding": cfg.data.encoding,
            "chunk_bytes": cfg.data.chunk_bytes,
            "overlap_bytes": cfg.data.overlap_bytes,
            "eval_fraction": cfg.data.eval_fraction,
            "eval_longdoc_min_bytes": cfg.data.eval_longdoc_min_bytes,
            "eval_longdoc_fraction": cfg.data.eval_longdoc_fraction,
            "eval_split_strategy": cfg.data.eval_split_strategy,
            "replay_stage_a_fraction": cfg.data.replay_stage_a_fraction,
            "replay_stage_b_fraction": cfg.data.replay_stage_b_fraction,
            "replay_stage_c_fraction": cfg.data.replay_stage_c_fraction,
            "train_replay_stage_a_fraction": cfg.data.train_replay_stage_a_fraction,
            "train_replay_stage_b_fraction": cfg.data.train_replay_stage_b_fraction,
            "train_replay_stage_c_fraction": cfg.data.train_replay_stage_c_fraction,
            "eval_replay_stage_a_fraction": cfg.data.eval_replay_stage_a_fraction,
            "eval_replay_stage_b_fraction": cfg.data.eval_replay_stage_b_fraction,
            "eval_replay_stage_c_fraction": cfg.data.eval_replay_stage_c_fraction,
            "seed": cfg.data.seed,
        },
        "input": {
            "preserve_document_boundaries": cfg.input.preserve_document_boundaries,
            "normalize_newlines": cfg.input.normalize_newlines,
        },
        "output": {
            "output_dir": str(run_dir / "output"),
            "format": cfg.output.format,
            "compression": cfg.output.compression,
            "max_records_per_shard": cfg.output.max_records_per_shard,
            "shard_prefix": cfg.output.shard_prefix,
            "resume": False,
            "resume_policy": "strict",
            "dry_run": bool(args.dry_run),
            "dry_run_max_records": max(1, min(64, args.records)),
            "log_token_lengths": cfg.output.log_token_lengths,
            "log_byte_lengths": cfg.output.log_byte_lengths,
            "stop_after_stage": cfg.output.stop_after_stage,
        },
        "stage_a": {
            "enabled": cfg.stage_a.enabled,
            "teacher_name": cfg.stage_a.teacher_name,
            "mode": cfg.stage_a.mode,
            "top_k": cfg.stage_a.top_k,
            "temperature": cfg.stage_a.temperature,
            "model_name_or_path": cfg.stage_a.model_name_or_path,
            "device_map": cfg.stage_a.device_map,
            "hf_offload_layers": cfg.stage_a.hf_offload_layers,
            "torch_dtype": cfg.stage_a.torch_dtype,
            "max_context": cfg.stage_a.max_context,
            "batch_size": cfg.stage_a.batch_size,
            "tensor_parallel_size": cfg.stage_a.tensor_parallel_size,
            "dtype": cfg.stage_a.dtype,
            "gpu_memory_utilization": cfg.stage_a.gpu_memory_utilization,
            "trust_remote_code": cfg.stage_a.trust_remote_code,
            "extract_hidden_summary": cfg.stage_a.extract_hidden_summary,
        },
        "stage_b": {
            "enabled": cfg.stage_b.enabled,
            "teacher_name": cfg.stage_b.teacher_name,
            "mode": cfg.stage_b.mode,
            "top_k": cfg.stage_b.top_k,
            "temperature": cfg.stage_b.temperature,
            "context_window": cfg.stage_b.context_window,
            "stride": cfg.stage_b.stride,
            "window_policy": cfg.stage_b.window_policy,
            "max_teacher_context": cfg.stage_b.max_teacher_context,
            "target_region_policy": cfg.stage_b.target_region_policy,
            "extract_hidden_summary": cfg.stage_b.extract_hidden_summary,
        },
        "stage_c": {
            "enabled": cfg.stage_c.enabled,
            "teacher_name": cfg.stage_c.teacher_name,
            "mode": cfg.stage_c.mode,
            "top_k": cfg.stage_c.top_k,
            "temperature": cfg.stage_c.temperature,
            "task_type": cfg.stage_c.task_type,
            "template_name": cfg.stage_c.template_name,
            "template_kwargs": cfg.stage_c.template_kwargs,
            "deterministic": cfg.stage_c.deterministic,
            "extract_hidden_summary": cfg.stage_c.extract_hidden_summary,
        },
    }

    benchmark_config = run_dir / "benchmark_config.toml"
    _write_toml(benchmark_config, bench_cfg)

    summary = run_pipeline(str(benchmark_config))
    train_records = len(read_jsonl_records(summary["train_path"]))
    eval_records = len(read_jsonl_records(summary["eval_path"]))
    output_records = train_records + eval_records

    timing = summary.get("timing", {}) if isinstance(summary, dict) else {}
    total_seconds = float(timing.get("total_runtime_seconds", 0.0) or 0.0)
    records_per_sec = (output_records / total_seconds) if total_seconds > 0 else 0.0

    print("Pipeline benchmark summary")
    print(f"  run_dir: {run_dir}")
    print(f"  config: {benchmark_config}")
    print(f"  selected_input_docs: {subset_count}")
    print(f"  output_records: {output_records}")
    print(f"  skipped_records: {summary.get('skipped_records', 0)}")
    print(f"  records_per_second: {records_per_sec:.4f}")
    for line in format_timing_report(summary):
        print(f"  {line}")


if __name__ == "__main__":
    main()
