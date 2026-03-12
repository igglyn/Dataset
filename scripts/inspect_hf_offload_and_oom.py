"""Inspect HF device-map placement and probe small-batch OOM risk.

Usage:
  python scripts/inspect_hf_offload_and_oom.py --model distilgpt2 --device-map auto --batch-sizes 1 2 4
  python scripts/inspect_hf_offload_and_oom.py --config configs/examples/hf_backend.toml --samples 8
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from distill_factory.config.schema import load_config
from distill_factory.teachers.hf_causal_lm import HFCausalLMTeacher


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect HF offload placement and run OOM probe batches")
    p.add_argument("--config", type=Path, default=None, help="Optional distill config TOML to source HF settings from")
    p.add_argument("--model", default="distilgpt2", help="HF model name/path (ignored when --config is set)")
    p.add_argument("--device-map", default="auto", help="HF device_map value (ignored when --config is set)")
    p.add_argument("--torch-dtype", default="float16", help="HF torch dtype (ignored when --config is set)")
    p.add_argument("--max-context", type=int, default=1024, help="Max context (ignored when --config is set)")
    p.add_argument("--hf-offload-layers", type=int, default=None, help="Optional number of trailing layers to offload to CPU")
    p.add_argument("--samples", type=int, default=4, help="Rows to run per batch-size probe")
    p.add_argument("--seq-len", type=int, default=128, help="Approx tokenizable prompt length")
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4], help="Batch sizes to probe")
    return p.parse_args()


def _teacher_from_args(args: argparse.Namespace) -> HFCausalLMTeacher:
    if args.config is not None:
        cfg = load_config(args.config)
        return HFCausalLMTeacher(
            model_name_or_path=cfg.stage_a.model_name_or_path,
            device_map=cfg.stage_a.device_map,
            torch_dtype=cfg.stage_a.torch_dtype,
            max_context=cfg.stage_a.max_context,
            batch_size=cfg.stage_a.batch_size,
            hf_pad_token_id=cfg.stage_a.hf_pad_token_id,
            hf_offload_layers=cfg.stage_a.hf_offload_layers,
        )
    return HFCausalLMTeacher(
        model_name_or_path=args.model,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        max_context=args.max_context,
        batch_size=max(1, int(args.batch_sizes[0])),
        hf_offload_layers=args.hf_offload_layers,
    )


def _extract_hf_map(model: Any) -> dict[str, str] | str | None:
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict):
        return {str(k): str(v) for k, v in device_map.items()}
    if device_map is None:
        return None
    return str(device_map)


def main() -> None:
    args = _parse_args()
    teacher = _teacher_from_args(args)

    report: dict[str, Any] = {
        "model": teacher.model_name_or_path,
        "requested_device_map": teacher.device_map,
        "hf_offload_layers": teacher.hf_offload_layers,
        "torch_dtype": teacher.torch_dtype,
        "oom_probe": [],
    }

    try:
        teacher.prepare()
        report["resolved_device_map_input"] = teacher._resolve_device_map()
        report["loaded_hf_device_map"] = _extract_hf_map(teacher._model)

        base_prompt = "x " * max(16, int(args.seq_len))
        for batch_size in [max(1, int(b)) for b in args.batch_sizes]:
            records = [{"raw_bytes": base_prompt.encode("utf-8"), "top_k": 4} for _ in range(batch_size)]
            row: dict[str, Any] = {"batch_size": batch_size}
            started = time.perf_counter()
            try:
                outs = teacher.infer_topk(records[: int(args.samples)])
                row["ok"] = True
                row["elapsed_s"] = round(time.perf_counter() - started, 4)
                row["outputs"] = len(outs)
            except RuntimeError as exc:
                msg = str(exc)
                is_oom = "out of memory" in msg.lower() or "cuda oom" in msg.lower()
                row["ok"] = False
                row["oom"] = bool(is_oom)
                row["error"] = msg
                row["elapsed_s"] = round(time.perf_counter() - started, 4)
            report["oom_probe"].append(row)
    except ModuleNotFoundError as exc:
        report["ok"] = False
        report["error"] = str(exc)
    finally:
        teacher.close()

    print(json.dumps(report, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
