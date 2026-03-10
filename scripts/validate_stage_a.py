#!/usr/bin/env python3
"""Run a tiny Stage A dense validation job before large-scale dataset builds.

For selection-aware validation modes, use scripts/validate_stage_a_selection.py.
"""

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


def _toml_literal(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, list):
        inner = ", ".join(_toml_literal(v) for v in value)
        return f"[{inner}]"
    if isinstance(value, dict):
        parts = [f"{k} = {_toml_literal(v)}" for k, v in value.items()]
        return "{ " + ", ".join(parts) + " }"
    raise TypeError(f"Unsupported TOML type for {value!r}")


def _write_toml(path: Path, cfg: dict[str, Any]) -> None:
    lines: list[str] = []
    for section in ("data", "input", "output", "stage_a", "stage_b", "stage_c"):
        values = cfg.get(section, {})
        lines.append(f"[{section}]")
        for key, value in values.items():
            lines.append(f"{key} = {_toml_literal(value)}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _copy_validation_inputs(input_path: Path, file_glob: str, limit: int, dest_root: Path) -> tuple[Path, int]:
    files = sorted([p for p in input_path.glob(file_glob) if p.is_file()], key=lambda p: p.as_posix())
    selected = files[:limit]
    if not selected:
        raise ValueError(f"No input files matched {file_glob!r} under {input_path}")

    out_input = dest_root / "validation_input"
    out_input.mkdir(parents=True, exist_ok=True)
    for src in selected:
        target = out_input / src.name
        if target.exists():
            stem = src.stem
            suffix = src.suffix
            i = 1
            while True:
                candidate = out_input / f"{stem}_{i}{suffix}"
                if not candidate.exists():
                    target = candidate
                    break
                i += 1
        shutil.copy2(src, target)
    return out_input, len(selected)


def _average_entropy(records: list[dict[str, Any]]) -> float | None:
    vals = [float(r["entropy"]) for r in records if r.get("entropy") is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _print_summary(summary: dict[str, Any], output_records: list[dict[str, Any]], teacher_name: str, used_fallback: bool) -> None:
    avg_entropy = _average_entropy(output_records)
    failed = int(summary.get("skipped_records", 0)) > 0

    print("Stage A validation summary")
    print(f"  input_docs: {summary.get('doc_count')}")
    print(f"  chunks: {summary.get('chunk_count')}")
    print(f"  output_records: {len(output_records)}")
    print(f"  teacher_name: {teacher_name}")
    print(f"  used_dummy_fallback: {used_fallback}")
    print(f"  average_entropy: {'n/a' if avg_entropy is None else f'{avg_entropy:.4f}'}")
    print(f"  any_records_failed: {failed}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small Stage A bring-up validation path.")
    parser.add_argument("--config", default="configs/examples/default.toml", help="Base config path.")
    parser.add_argument("--input-path", default=None, help="Optional override for [data].input_path.")
    parser.add_argument("--file-glob", default=None, help="Optional override for [data].file_glob.")
    parser.add_argument("--records", type=int, default=16, help="Small number of input docs to validate (8-32 recommended).")
    parser.add_argument(
        "--output-root",
        default="data/validation",
        help="Root directory for validation outputs.",
    )
    args = parser.parse_args()

    if args.records <= 0:
        raise ValueError("--records must be > 0")

    cfg = load_config(args.config)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.output_root) / f"stage_a_validation_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input_path) if args.input_path is not None else Path(cfg.data.input_path)
    file_glob = args.file_glob if args.file_glob is not None else cfg.data.file_glob

    validation_input, selected_docs = _copy_validation_inputs(
        input_path,
        file_glob,
        limit=args.records,
        dest_root=run_dir,
    )

    config_data: dict[str, Any] = {
        "data": {
            "input_path": str(validation_input),
            "file_glob": file_glob,
            "encoding": cfg.data.encoding,
            "chunk_bytes": cfg.data.chunk_bytes,
            "overlap_bytes": cfg.data.overlap_bytes,
            "eval_fraction": cfg.data.eval_fraction,
            "eval_longdoc_min_bytes": cfg.data.eval_longdoc_min_bytes,
            "eval_longdoc_fraction": 0.0,
            "eval_split_strategy": cfg.data.eval_split_strategy,
            "replay_stage_a_fraction": 1.0,
            "replay_stage_b_fraction": 0.0,
            "replay_stage_c_fraction": 0.0,
            "train_replay_stage_a_fraction": 1.0,
            "train_replay_stage_b_fraction": 0.0,
            "train_replay_stage_c_fraction": 0.0,
            "eval_replay_stage_a_fraction": 1.0,
            "eval_replay_stage_b_fraction": 0.0,
            "eval_replay_stage_c_fraction": 0.0,
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
            "dry_run": False,
            "dry_run_max_records": max(1, min(32, args.records)),
            "log_token_lengths": cfg.output.log_token_lengths,
            "log_byte_lengths": cfg.output.log_byte_lengths,
            "stop_after_stage": "stage_a",
        },
        "stage_a": {
            "enabled": True,
            "teacher_name": cfg.stage_a.teacher_name,
            "backend_type": cfg.stage_a.backend_type,
            "mode": cfg.stage_a.mode,
            "top_k": cfg.stage_a.top_k,
            "temperature": cfg.stage_a.temperature,
            "model_name_or_path": cfg.stage_a.model_name_or_path,
            "device_map": cfg.stage_a.device_map,
            "torch_dtype": cfg.stage_a.torch_dtype,
            "max_context": cfg.stage_a.max_context,
            "batch_size": cfg.stage_a.batch_size,
            "tensor_parallel_size": cfg.stage_a.tensor_parallel_size,
            "dtype": cfg.stage_a.dtype,
            "gpu_memory_utilization": cfg.stage_a.gpu_memory_utilization,
            "trust_remote_code": cfg.stage_a.trust_remote_code,
            "llama_base_url": cfg.stage_a.llama_base_url,
            "llama_model_hint": cfg.stage_a.llama_model_hint,
            "llama_request_timeout": cfg.stage_a.llama_request_timeout,
            "extract_hidden_summary": cfg.stage_a.extract_hidden_summary,
            "enable_position_filtering": cfg.stage_a.enable_position_filtering,
            "entropy_threshold": cfg.stage_a.entropy_threshold,
            "top1_gap_threshold": cfg.stage_a.top1_gap_threshold,
            "selection_window_radius": cfg.stage_a.selection_window_radius,
            "selection_mode": cfg.stage_a.selection_mode,
            "minimum_selected_positions_per_record": cfg.stage_a.minimum_selected_positions_per_record,
        },
        "stage_b": {
            "enabled": False,
            "teacher_name": cfg.stage_b.teacher_name,
            "backend_type": cfg.stage_b.backend_type,
            "mode": cfg.stage_b.mode,
            "top_k": cfg.stage_b.top_k,
            "temperature": cfg.stage_b.temperature,
            "context_window": cfg.stage_b.context_window,
            "stride": cfg.stage_b.stride,
            "window_policy": cfg.stage_b.window_policy,
            "max_teacher_context": cfg.stage_b.max_teacher_context,
            "target_region_policy": cfg.stage_b.target_region_policy,
            "llama_base_url": cfg.stage_b.llama_base_url,
            "llama_model_hint": cfg.stage_b.llama_model_hint,
            "llama_request_timeout": cfg.stage_b.llama_request_timeout,
            "extract_hidden_summary": cfg.stage_b.extract_hidden_summary,
            "enable_position_filtering": cfg.stage_b.enable_position_filtering,
            "entropy_threshold": cfg.stage_b.entropy_threshold,
            "top1_gap_threshold": cfg.stage_b.top1_gap_threshold,
            "selection_window_radius": cfg.stage_b.selection_window_radius,
            "selection_mode": "none",
            "minimum_selected_positions_per_record": cfg.stage_b.minimum_selected_positions_per_record,
        },
        "stage_c": {
            "enabled": False,
            "teacher_name": cfg.stage_c.teacher_name,
            "backend_type": cfg.stage_c.backend_type,
            "mode": cfg.stage_c.mode,
            "top_k": cfg.stage_c.top_k,
            "temperature": cfg.stage_c.temperature,
            "task_type": cfg.stage_c.task_type,
            "template_name": cfg.stage_c.template_name,
            "template_kwargs": cfg.stage_c.template_kwargs,
            "deterministic": cfg.stage_c.deterministic,
            "llama_base_url": cfg.stage_c.llama_base_url,
            "llama_model_hint": cfg.stage_c.llama_model_hint,
            "llama_request_timeout": cfg.stage_c.llama_request_timeout,
            "extract_hidden_summary": cfg.stage_c.extract_hidden_summary,
        },
    }

    validation_config = run_dir / "validation_stage_a.toml"
    _write_toml(validation_config, config_data)

    used_teacher = str(config_data["stage_a"]["teacher_name"])
    used_fallback = False

    try:
        summary = run_pipeline(str(validation_config))
    except Exception as exc:
        print(f"Primary teacher run failed for '{used_teacher}': {exc}")
        print("Falling back to DummyTeacher for validation run.")
        config_data["stage_a"]["teacher_name"] = "dummy"
        used_teacher = "dummy"
        used_fallback = True
        _write_toml(validation_config, config_data)
        summary = run_pipeline(str(validation_config))

    train_records = read_jsonl_records(summary["train_path"])
    eval_records = read_jsonl_records(summary["eval_path"])
    combined = train_records + eval_records

    print(f"validation_run_dir: {run_dir}")
    print(f"validation_config: {validation_config}")
    print(f"selected_input_docs: {selected_docs}")
    _print_summary(summary, combined, teacher_name=used_teacher, used_fallback=used_fallback)


if __name__ == "__main__":
    main()
