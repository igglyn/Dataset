#!/usr/bin/env python3
"""Build distilled dataset from raw documents using configured stages."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from distill_factory.config.schema import load_config
from distill_factory.data.formats import from_record
from distill_factory.data.splits import split_records_with_longdoc_eval
from distill_factory.pipeline.orchestrator import run_pipeline
from distill_factory.storage.reader import (
    merge_jsonl_records,
    merge_jsonl_records_with_ratios,
    read_jsonl,
    read_jsonl_records,
)
from distill_factory.storage.writer import write_jsonl


def _run_merge_mode(merge_inputs: list[str], merge_ratios: list[float] | None, merge_output: str) -> None:
    if merge_ratios is None:
        merged = merge_jsonl_records(merge_inputs)
    else:
        if len(merge_ratios) != len(merge_inputs):
            raise ValueError("--merge-ratios length must match --merge-inputs length")
        merged = merge_jsonl_records_with_ratios(list(zip(merge_inputs, merge_ratios)))

    samples = [from_record(r) for r in merged]
    out_path = Path(merge_output)
    write_jsonl(samples, out_path)
    print(f"Merge complete: records={len(samples)} output={out_path}")


def _apply_longdoc_eval_split_if_needed(config_path: str, summary: dict[str, object]) -> dict[str, object]:
    cfg = load_config(config_path)
    if float(cfg.data.eval_longdoc_fraction) <= 0.0:
        return summary

    train_path = Path(str(summary["train_path"]))
    eval_path = Path(str(summary["eval_path"]))
    eval_longdoc_path = train_path.parent / f"eval_longdoc.{train_path.suffix.lstrip('.')}"

    combined = read_jsonl_records(train_path) + read_jsonl_records(eval_path)
    train_records, eval_records, eval_longdoc_records = split_records_with_longdoc_eval(
        combined,
        eval_fraction=float(cfg.data.eval_fraction),
        eval_longdoc_fraction=float(cfg.data.eval_longdoc_fraction),
        eval_longdoc_min_bytes=int(cfg.data.eval_longdoc_min_bytes),
        eval_split_strategy=str(cfg.data.eval_split_strategy),
        seed=int(cfg.data.seed),
    )

    train_samples = [from_record(r) for r in train_records]
    eval_samples = [from_record(r) for r in eval_records]
    eval_longdoc_samples = [from_record(r) for r in eval_longdoc_records]

    write_jsonl(train_samples, train_path)
    write_jsonl(eval_samples, eval_path)
    write_jsonl(eval_longdoc_samples, eval_longdoc_path)

    return {
        **summary,
        "train_count": len(train_samples),
        "eval_count": len(eval_samples),
        "eval_longdoc_count": len(eval_longdoc_samples),
        "eval_longdoc_path": str(eval_longdoc_path),
    }


def _apply_sharding_if_needed(config_path: str, summary: dict[str, object]) -> dict[str, object]:
    cfg = load_config(config_path)
    max_per = int(cfg.output.max_records_per_shard)
    if max_per <= 0:
        return summary

    train_path = Path(str(summary["train_path"]))
    eval_path = Path(str(summary["eval_path"]))
    eval_longdoc_path = Path(str(summary["eval_longdoc_path"])) if summary.get("eval_longdoc_path") else None

    train_samples = read_jsonl(train_path)
    eval_samples = read_jsonl(eval_path)

    train_prefix = f"{cfg.output.shard_prefix}-train"
    eval_prefix = f"{cfg.output.shard_prefix}-eval"

    train_written = write_jsonl(
        train_samples,
        train_path,
        max_records_per_shard=max_per,
        shard_prefix=train_prefix,
    )
    eval_written = write_jsonl(
        eval_samples,
        eval_path,
        max_records_per_shard=max_per,
        shard_prefix=eval_prefix,
    )

    out: dict[str, object] = {
        **summary,
        "train_shards": train_written,
        "eval_shards": eval_written,
    }

    if eval_longdoc_path is not None and eval_longdoc_path.exists():
        eval_longdoc_samples = read_jsonl(eval_longdoc_path)
        eval_longdoc_prefix = f"{cfg.output.shard_prefix}-eval_longdoc"
        eval_longdoc_written = write_jsonl(
            eval_longdoc_samples,
            eval_longdoc_path,
            max_records_per_shard=max_per,
            shard_prefix=eval_longdoc_prefix,
        )
        out["eval_longdoc_shards"] = eval_longdoc_written

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build distillation dataset.")
    parser.add_argument("--config", help="Path to TOML config file.")
    parser.add_argument("--merge-inputs", nargs="*", help="Optional JSONL datasets to merge.")
    parser.add_argument(
        "--merge-ratios",
        nargs="*",
        type=float,
        help="Optional ratios for --merge-inputs (same length).",
    )
    parser.add_argument(
        "--merge-output",
        default="data/processed/merged.jsonl",
        help="Output path for merged JSONL.",
    )
    args = parser.parse_args()

    if args.merge_inputs:
        _run_merge_mode(args.merge_inputs, args.merge_ratios, args.merge_output)
        return

    if not args.config:
        raise ValueError("--config is required when not using --merge-inputs")

    summary = run_pipeline(args.config)
    summary = _apply_longdoc_eval_split_if_needed(args.config, summary)
    summary = _apply_sharding_if_needed(args.config, summary)

    print(
        "Build complete: "
        f"docs={summary['doc_count']} chunks={summary['chunk_count']} "
        f"train={summary['train_count']} eval={summary['eval_count']} eval_longdoc={summary.get('eval_longdoc_count', 0)} "
        f"train_path={summary['train_path']} eval_path={summary['eval_path']} eval_longdoc_path={summary.get('eval_longdoc_path')} "
        f"stage_a_teachers={summary.get('stage_a_teachers')} "
        f"stage_b_teachers={summary.get('stage_b_teachers')} "
        f"stage_c_teachers={summary.get('stage_c_teachers')} "
        f"train_shards={summary.get('train_shards')} eval_shards={summary.get('eval_shards')} eval_longdoc_shards={summary.get('eval_longdoc_shards')}"
    )


if __name__ == "__main__":
    main()
