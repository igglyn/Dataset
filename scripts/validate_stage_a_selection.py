#!/usr/bin/env python3
"""Tiny end-to-end Stage A selection-aware validation harness."""

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
from scripts.inspect_dataset import per_token_field_presence, selection_inspection


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
        for key, value in cfg.get(section, {}).items():
            if value is None:
                continue
            lines.append(f"{key} = {_toml_literal(value)}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _copy_inputs(input_path: Path, file_glob: str, limit: int, dest_root: Path) -> tuple[Path, int]:
    files = sorted([p for p in input_path.glob(file_glob) if p.is_file()], key=lambda p: p.as_posix())[:limit]
    if not files:
        raise ValueError(f"No input files matched {file_glob!r} under {input_path}")

    out_input = dest_root / "validation_input"
    out_input.mkdir(parents=True, exist_ok=True)
    for src in files:
        target = out_input / src.name
        if target.exists():
            target = out_input / f"{src.stem}_{len(list(out_input.glob(src.stem + '*')))}{src.suffix}"
        shutil.copy2(src, target)
    return out_input, len(files)


def _average(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def _avg_selected_window_length(records: list[dict[str, Any]]) -> float | None:
    lengths: list[float] = []
    for r in records:
        meta = r.get("extra_metadata")
        if not isinstance(meta, dict):
            continue
        if "selected_window_start" in meta and "selected_window_end" in meta:
            lengths.append(float(int(meta["selected_window_end"]) - int(meta["selected_window_start"]) + 1))
    return _average(lengths)


def _assert_required_per_token_fields(records: list[dict[str, Any]], *, require_entropy: bool, require_gap: bool) -> None:
    missing: list[str] = []
    if require_entropy:
        if any(r.get("per_token_entropy") is None for r in records):
            missing.append("per_token_entropy")
    if require_gap:
        if any(r.get("per_token_top1_gap") is None for r in records):
            missing.append("per_token_top1_gap")
    if missing:
        raise RuntimeError(
            "Selection is enabled but required per-token fields are missing in output records: "
            + ", ".join(missing)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Stage A selection-aware export with a tiny end-to-end run.")
    parser.add_argument("--config", default="configs/examples/default.toml", help="Base config path.")
    parser.add_argument("--mode", choices=["dense", "position_mask", "selected_windows"], default="dense")
    parser.add_argument("--records", type=int, default=8, help="Number of input docs to include in tiny run.")
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--file-glob", default=None)
    parser.add_argument("--output-root", default="data/validation")
    parser.add_argument("--entropy-threshold", type=float, default=0.8)
    parser.add_argument("--top1-gap-threshold", type=float, default=None)
    parser.add_argument("--selection-window-radius", type=int, default=0)
    parser.add_argument("--minimum-selected-positions", type=int, default=None)
    args = parser.parse_args()

    if args.records <= 0:
        raise ValueError("--records must be > 0")

    cfg = load_config(args.config)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.output_root) / f"stage_a_selection_{args.mode}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input_path) if args.input_path is not None else Path(cfg.data.input_path)
    file_glob = args.file_glob if args.file_glob is not None else cfg.data.file_glob
    tiny_input, _ = _copy_inputs(input_path, file_glob, args.records, run_dir)

    selection_enabled = args.mode != "dense"
    selection_mode = "none" if args.mode == "dense" else args.mode
    entropy_threshold = None if not selection_enabled else args.entropy_threshold
    top1_gap_threshold = None if not selection_enabled else args.top1_gap_threshold
    if selection_enabled and entropy_threshold is None and top1_gap_threshold is None:
        raise ValueError("Selection mode requires --entropy-threshold and/or --top1-gap-threshold")

    validation_cfg = {
        "data": {
            "input_path": str(tiny_input),
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
            "format": "jsonl",
            "compression": None,
            "max_records_per_shard": 0,
            "shard_prefix": "stage_a_selection",
            "resume": False,
            "resume_policy": "strict",
            "dry_run": False,
            "dry_run_max_records": max(1, min(16, args.records)),
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
            "extract_hidden_summary": cfg.stage_a.extract_hidden_summary,
            "enable_position_filtering": selection_enabled,
            "entropy_threshold": entropy_threshold,
            "top1_gap_threshold": top1_gap_threshold,
            "selection_window_radius": int(args.selection_window_radius),
            "selection_mode": selection_mode,
            "minimum_selected_positions_per_record": args.minimum_selected_positions,
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
        },
    }

    run_cfg_path = run_dir / "validate_stage_a_selection.toml"
    _write_toml(run_cfg_path, validation_cfg)
    summary = run_pipeline(str(run_cfg_path))

    records = read_jsonl_records(summary["train_path"]) + read_jsonl_records(summary["eval_path"])
    presence = per_token_field_presence(records)
    selection = selection_inspection(records)

    if selection_enabled:
        _assert_required_per_token_fields(
            records,
            require_entropy=entropy_threshold is not None,
            require_gap=top1_gap_threshold is not None,
        )

    avg_selected_positions = selection.get("average_selected_position_count")
    avg_window_len = _avg_selected_window_length(records)

    print("Stage A selection validation")
    print(f"  mode: {args.mode}")
    print(f"  input_record_count: {int(summary.get('chunk_count', 0))}")
    print(f"  output_record_count: {len(records)}")
    print(f"  per_token_fields_present: {presence}")
    print(f"  selection_metadata_present: {selection['selection_filtered_record_proportion'] > 0.0}")
    print(
        "  average_selected_position_count: "
        + ("n/a" if avg_selected_positions is None else f"{float(avg_selected_positions):.2f}")
    )
    print(
        "  average_selected_window_length: "
        + ("n/a" if avg_window_len is None else f"{float(avg_window_len):.2f}")
    )
    print(f"  run_config: {run_cfg_path}")
    print(f"  run_output_dir: {validation_cfg['output']['output_dir']}")


if __name__ == "__main__":
    main()
