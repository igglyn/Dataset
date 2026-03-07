"""Pipeline orchestration for stage A/B/C dataset generation."""

from __future__ import annotations

from pathlib import Path
from random import Random
from typing import Any

from distill_factory.config.schema import load_config
from distill_factory.data.chunking import chunk_documents
from distill_factory.data.formats import DistilledSample
from distill_factory.data.ingest import ingest_documents
from distill_factory.data.splits import split_records_by_doc_id
from distill_factory.pipeline.resume_state import (
    build_initial_resume_state,
    load_resume_state,
    validate_resume_state,
    write_resume_state,
)
from distill_factory.pipeline.stage_a import run_stage_a
from distill_factory.pipeline.stage_b import run_stage_b
from distill_factory.pipeline.stage_c import run_stage_c
from distill_factory.storage.reader import list_shard_paths, read_jsonl_records
from distill_factory.storage.writer import write_jsonl, write_parquet
from distill_factory.teachers.registry import get_teacher
from distill_factory.utils.hashing import deduplicate_records
from distill_factory.utils.logging import append_record_failure


_STAGE_ORDER = ("stage_a", "stage_b", "stage_c")


def _clone_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cloned: list[dict[str, Any]] = []
    for rec in records:
        c = dict(rec)
        if isinstance(rec.get("extra_metadata"), dict):
            c["extra_metadata"] = dict(rec["extra_metadata"])
        cloned.append(c)
    return cloned


def _allocate_counts(total: int, ratios: list[float]) -> list[int]:
    if total <= 0 or not ratios:
        return [0 for _ in ratios]
    ratio_sum = sum(ratios)
    if ratio_sum <= 0:
        return [0 for _ in ratios]

    raw = [total * (r / ratio_sum) for r in ratios]
    counts = [int(x) for x in raw]
    remainder = total - sum(counts)
    fracs = sorted([(raw[i] - counts[i], i) for i in range(len(raw))], reverse=True)
    for _, idx in fracs[:remainder]:
        counts[idx] += 1
    return counts


def _deterministic_subsample(records: list[dict[str, Any]], k: int, seed: int) -> list[dict[str, Any]]:
    if k <= 0:
        return []
    if k >= len(records):
        return records

    keyed = sorted(
        records,
        key=lambda r: (str(r.get("doc_id", "")), int(r.get("chunk_index", -1)), int(r.get("byte_start", -1))),
    )
    rng = Random(seed)
    idx = list(range(len(keyed)))
    rng.shuffle(idx)
    chosen = sorted(idx[:k])
    return [keyed[i] for i in chosen]


def _apply_stage_mixture(
    records: list[dict[str, Any]],
    stage_name: str,
    mode: str,
    mixture: list[Any],
    cfg: Any,
    seed: int,
    dry_run: bool = False,
    failure_output_dir: Path | None = None,
    skip_stats: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    if not records:
        return records

    teacher_names = [m.teacher_name for m in mixture]
    ratios = [float(m.ratio) for m in mixture]
    target_counts = _allocate_counts(len(records), ratios)

    mixed: list[dict[str, Any]] = []
    stage_attempted = 0
    stage_failed = 0

    for i, teacher_name in enumerate(teacher_names):
        stage_in = _clone_records(records)

        stage_out: list[dict[str, Any]] = []
        for record in stage_in:
            stage_attempted += 1
            try:
                single = [record]
                if stage_name == "stage_a":
                    out = run_stage_a(single, teacher_name=teacher_name, mode=mode, dry_run=dry_run)
                elif stage_name == "stage_b":
                    out = run_stage_b(
                        single,
                        teacher_name=teacher_name,
                        mode=mode,
                        context_window=cfg.stage_b.context_window,
                        stride=cfg.stage_b.stride,
                        max_teacher_context=cfg.stage_b.max_teacher_context,
                        window_policy=cfg.stage_b.window_policy,
                        target_region_policy=cfg.stage_b.target_region_policy,
                        dry_run=dry_run,
                    )
                elif stage_name == "stage_c":
                    out = run_stage_c(
                        single,
                        teacher_name=teacher_name,
                        mode=mode,
                        template_name=cfg.stage_c.template_name,
                        template_kwargs=cfg.stage_c.template_kwargs,
                        deterministic=cfg.stage_c.deterministic,
                        dry_run=dry_run,
                    )
                else:
                    raise ValueError(f"Unknown stage_name: {stage_name}")
                if out:
                    stage_out.extend(out)
            except Exception as exc:
                stage_failed += 1
                if skip_stats is not None:
                    skip_stats["skipped_records"] = int(skip_stats.get("skipped_records", 0)) + 1
                if failure_output_dir is not None:
                    append_record_failure(
                        failure_output_dir,
                        stage_name=stage_name,
                        teacher_name=teacher_name,
                        doc_id=str(record.get("doc_id")) if record.get("doc_id") is not None else None,
                        chunk_index=int(record.get("chunk_index")) if record.get("chunk_index") is not None else None,
                        error_message=str(exc),
                    )

        take = _deterministic_subsample(stage_out, target_counts[i], seed=seed + i + 1)
        mixed.extend(take)

    if stage_attempted > 0 and stage_failed == stage_attempted:
        raise RuntimeError(
            f"Fatal stage-level error in {stage_name}: all records failed for configured teachers. "
            f"Check failure log for details."
        )

    return mixed


def _run_enabled_stages_with_history(
    records: list[dict[str, Any]],
    cfg: Any,
    dry_run: bool = False,
    failure_output_dir: Path | None = None,
    skip_stats: dict[str, int] | None = None,
    stop_after_stage: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    out = records
    history: dict[str, list[dict[str, Any]]] = {}

    if cfg.stage_a.enabled:
        out = _apply_stage_mixture(
            out,
            stage_name="stage_a",
            mode=cfg.stage_a.mode,
            mixture=cfg.stage_a.teacher_mixture,
            cfg=cfg,
            seed=cfg.data.seed + 101,
            dry_run=dry_run,
            failure_output_dir=failure_output_dir,
            skip_stats=skip_stats,
        )
        history["stage_a"] = _clone_records(out)
        if stop_after_stage == "stage_a":
            return out, history

    if cfg.stage_b.enabled:
        out = _apply_stage_mixture(
            out,
            stage_name="stage_b",
            mode=cfg.stage_b.mode,
            mixture=cfg.stage_b.teacher_mixture,
            cfg=cfg,
            seed=cfg.data.seed + 202,
            dry_run=dry_run,
            failure_output_dir=failure_output_dir,
            skip_stats=skip_stats,
        )
        history["stage_b"] = _clone_records(out)
        if stop_after_stage == "stage_b":
            return out, history

    if cfg.stage_c.enabled:
        out = _apply_stage_mixture(
            out,
            stage_name="stage_c",
            mode=cfg.stage_c.mode,
            mixture=cfg.stage_c.teacher_mixture,
            cfg=cfg,
            seed=cfg.data.seed + 303,
            dry_run=dry_run,
            failure_output_dir=failure_output_dir,
            skip_stats=skip_stats,
        )
        history["stage_c"] = _clone_records(out)
        if stop_after_stage == "stage_c":
            return out, history

    return out, history


def _apply_static_curriculum_mix(
    *,
    final_records: list[dict[str, Any]],
    stage_history: dict[str, list[dict[str, Any]]],
    stage_fractions: dict[str, float],
    seed: int,
) -> list[dict[str, Any]]:
    available_stages = [s for s in _STAGE_ORDER if s in stage_history]
    if len(available_stages) <= 1:
        return final_records

    ratio_stages = [s for s in available_stages if stage_fractions.get(s, 0.0) > 0.0]
    if not ratio_stages:
        return final_records

    target_total = len(final_records)
    ratios = [float(stage_fractions[s]) for s in ratio_stages]
    counts = _allocate_counts(target_total, ratios)

    mixed: list[dict[str, Any]] = []
    for idx, stage_name in enumerate(ratio_stages):
        stage_records = _clone_records(stage_history.get(stage_name, []))
        selected = _deterministic_subsample(stage_records, counts[idx], seed=seed + (idx + 1) * 17)
        mixed.extend(selected)

    return mixed if mixed else final_records


def _to_distilled_samples(records: list[dict[str, Any]]) -> list[DistilledSample]:
    samples: list[DistilledSample] = []
    for rec in records:
        samples.append(
            DistilledSample(
                doc_id=str(rec["doc_id"]),
                chunk_index=int(rec["chunk_index"]),
                byte_start=int(rec["byte_start"]),
                byte_end=int(rec["byte_end"]),
                raw_bytes=bytes(rec["raw_bytes"]),
                split=str(rec["split"]),
                teacher_name=str(rec.get("teacher_name", "unassigned_teacher")),
                stage_name=str(rec.get("stage_name", "unassigned_stage")),
                mode=str(rec.get("mode", "unknown")),
                top_k_ids=rec.get("top_k_ids"),
                top_k_logprobs=rec.get("top_k_logprobs"),
                entropy=rec.get("entropy"),
                structured_output=rec.get("structured_output"),
                extra_metadata=rec.get("extra_metadata"),
            )
        )
    return samples



def _run_teacher_startup_self_checks(cfg: Any) -> list[dict[str, Any]]:
    """Run lightweight teacher backend checks before expensive pipeline work."""
    checks: list[dict[str, Any]] = []

    stage_specs: list[tuple[str, bool, str, int, list[Any]]] = [
        ("stage_a", bool(cfg.stage_a.enabled), str(cfg.stage_a.mode), int(cfg.stage_a.top_k), list(cfg.stage_a.teacher_mixture)),
        ("stage_b", bool(cfg.stage_b.enabled), str(cfg.stage_b.mode), int(cfg.stage_b.top_k), list(cfg.stage_b.teacher_mixture)),
        ("stage_c", bool(cfg.stage_c.enabled), str(cfg.stage_c.mode), int(cfg.stage_c.top_k), list(cfg.stage_c.teacher_mixture)),
    ]

    seen: set[tuple[str, str, str, int]] = set()
    for stage_name, enabled, mode, top_k, mixture in stage_specs:
        if not enabled:
            continue
        if top_k < 1:
            raise ValueError(f"Invalid top_k for {stage_name}: expected >= 1, got {top_k}.")
        for mix in mixture:
            teacher_name = str(mix.teacher_name)
            key = (stage_name, teacher_name, mode, top_k)
            if key in seen:
                continue
            seen.add(key)

            teacher = get_teacher(teacher_name)
            try:
                check_fn = getattr(teacher, "startup_self_check", None)
                if callable(check_fn):
                    result = check_fn(requested_top_k=top_k)
                else:
                    teacher.prepare()
                    result = {"backend": teacher_name, "ok": True, "note": "startup_self_check not implemented"}
            except Exception as exc:
                raise RuntimeError(
                    f"Teacher startup self-check failed for stage '{stage_name}' with teacher '{teacher_name}' "
                    f"(mode={mode}, top_k={top_k}): {exc}"
                ) from exc
            finally:
                try:
                    teacher.close()
                except Exception:
                    pass

            checks.append({
                "stage_name": stage_name,
                "teacher_name": teacher_name,
                "mode": mode,
                "top_k": top_k,
                "result": result,
            })

    return checks

def _parse_shard_ids(paths: list[str]) -> list[str]:
    shard_ids: list[str] = []
    for path in paths:
        stem = Path(path).stem
        shard_ids.append(stem.split("-")[-1] if "-" in stem else stem)
    return sorted(set(shard_ids))


def _write_split(
    records: list[dict[str, Any]],
    output_path: Path,
    output_format: str,
    *,
    max_records_per_shard: int,
    shard_prefix: str,
    resume: bool,
    skipped_records_count: int = 0,
) -> tuple[int, int, list[str]]:
    unique_records, duplicates_skipped = deduplicate_records(records)
    samples = _to_distilled_samples(unique_records)

    if output_format == "jsonl":
        written_paths = write_jsonl(
            samples,
            output_path,
            max_records_per_shard=max_records_per_shard,
            shard_prefix=shard_prefix,
            append=resume,
            skipped_records_count=skipped_records_count,
        )
        total_after_write = len(read_jsonl_records(output_path))
    else:
        write_parquet(samples, output_path)
        written_paths = [str(output_path)]
        total_after_write = len(unique_records)

    return total_after_write, duplicates_skipped, written_paths


def run_pipeline(config_path: str) -> dict[str, Any]:
    """Run full pipeline: config -> ingest -> chunk -> split -> stages -> write."""
    cfg = load_config(config_path)

    output_dir = Path(cfg.output.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    startup_checks = _run_teacher_startup_self_checks(cfg)

    teacher_names = {
        "stage_a": [m.teacher_name for m in cfg.stage_a.teacher_mixture],
        "stage_b": [m.teacher_name for m in cfg.stage_b.teacher_mixture],
        "stage_c": [m.teacher_name for m in cfg.stage_c.teacher_mixture],
    }

    resume_state: dict[str, Any] | None = None
    resume_warning: str | None = None
    skip_stats: dict[str, int] = {"skipped_records": 0}
    if cfg.output.resume:
        resume_state = load_resume_state(output_dir)
        if resume_state is None:
            resume_state = build_initial_resume_state(cfg, config_path, teacher_names)
            write_resume_state(output_dir, resume_state)
        ok, message = validate_resume_state(resume_state, cfg, cfg.output.resume_policy)
        if not ok:
            raise ValueError(message or "resume state is incompatible with current config")
        resume_warning = message

    docs = ingest_documents(
        input_path=cfg.data.input_path,
        file_glob=cfg.data.file_glob,
        encoding=cfg.data.encoding,
        normalize_newlines=cfg.input.normalize_newlines,
    )

    chunks = chunk_documents(
        documents=docs,
        chunk_bytes=cfg.data.chunk_bytes,
        overlap_bytes=cfg.data.overlap_bytes,
        encoding=cfg.data.encoding,
    )

    train_records, eval_records = split_records_by_doc_id(
        chunks,
        eval_fraction=cfg.data.eval_fraction,
        seed=cfg.data.seed,
        method="shuffle",
    )

    for record in train_records:
        record["split"] = "train"
    for record in eval_records:
        record["split"] = "eval"

    if cfg.output.dry_run:
        train_records = _deterministic_subsample(train_records, cfg.output.dry_run_max_records, seed=cfg.data.seed + 1)
        eval_records = _deterministic_subsample(eval_records, cfg.output.dry_run_max_records, seed=cfg.data.seed + 2)

    train_final, train_history = _run_enabled_stages_with_history(
        train_records,
        cfg,
        dry_run=cfg.output.dry_run,
        failure_output_dir=output_dir,
        skip_stats=skip_stats,
        stop_after_stage=cfg.output.stop_after_stage,
    )
    eval_final, eval_history = _run_enabled_stages_with_history(
        eval_records,
        cfg,
        dry_run=cfg.output.dry_run,
        failure_output_dir=output_dir,
        skip_stats=skip_stats,
        stop_after_stage=cfg.output.stop_after_stage,
    )

    completed_stage_names = [stage for stage in _STAGE_ORDER if stage in train_history or stage in eval_history]
    if resume_state is not None:
        resume_state["completed_stages"] = completed_stage_names
        write_resume_state(output_dir, resume_state)

    train_stage_fractions = {
        "stage_a": cfg.data.train_replay_stage_a_fraction,
        "stage_b": cfg.data.train_replay_stage_b_fraction,
        "stage_c": cfg.data.train_replay_stage_c_fraction,
    }
    eval_stage_fractions = {
        "stage_a": cfg.data.eval_replay_stage_a_fraction,
        "stage_b": cfg.data.eval_replay_stage_b_fraction,
        "stage_c": cfg.data.eval_replay_stage_c_fraction,
    }

    train_records = _apply_static_curriculum_mix(
        final_records=train_final,
        stage_history=train_history,
        stage_fractions=train_stage_fractions,
        seed=cfg.data.seed + 401,
    )
    eval_records = _apply_static_curriculum_mix(
        final_records=eval_final,
        stage_history=eval_history,
        stage_fractions=eval_stage_fractions,
        seed=cfg.data.seed + 402,
    )

    ext = "jsonl" if cfg.output.format == "jsonl" else "parquet"
    train_path = output_dir / f"train.{ext}"
    eval_path = output_dir / f"eval.{ext}"

    train_written = 0
    eval_written = 0
    train_dupes = 0
    eval_dupes = 0

    train_done = bool((resume_state or {}).get("split_progress", {}).get("train", {}).get("completed", False))
    eval_done = bool((resume_state or {}).get("split_progress", {}).get("eval", {}).get("completed", False))

    if cfg.output.resume and train_done:
        train_written = int((resume_state or {}).get("split_progress", {}).get("train", {}).get("record_count", 0))
    else:
        train_written, train_dupes, train_written_paths = _write_split(
            train_records,
            train_path,
            cfg.output.format,
            max_records_per_shard=cfg.output.max_records_per_shard,
            shard_prefix=f"train-{cfg.output.shard_prefix}",
            resume=cfg.output.resume,
            skipped_records_count=0,
        )
        if resume_state is not None:
            progress = dict(resume_state.get("split_progress", {}))
            progress["train"] = {
                "completed": True,
                "record_count": train_written,
                "shard_ids": _parse_shard_ids(list_shard_paths(train_path)),
                "written_paths": train_written_paths,
            }
            resume_state["split_progress"] = progress
            write_resume_state(output_dir, resume_state)

    if cfg.output.resume and eval_done:
        eval_written = int((resume_state or {}).get("split_progress", {}).get("eval", {}).get("record_count", 0))
    else:
        eval_written, eval_dupes, eval_written_paths = _write_split(
            eval_records,
            eval_path,
            cfg.output.format,
            max_records_per_shard=cfg.output.max_records_per_shard,
            shard_prefix=f"eval-{cfg.output.shard_prefix}",
            resume=cfg.output.resume,
            skipped_records_count=skip_stats.get("skipped_records", 0),
        )
        if resume_state is not None:
            progress = dict(resume_state.get("split_progress", {}))
            progress["eval"] = {
                "completed": True,
                "record_count": eval_written,
                "shard_ids": _parse_shard_ids(list_shard_paths(eval_path)),
                "written_paths": eval_written_paths,
            }
            resume_state["split_progress"] = progress
            write_resume_state(output_dir, resume_state)

    return {
        "train_path": str(train_path),
        "eval_path": str(eval_path),
        "train_count": train_written,
        "eval_count": eval_written,
        "train_duplicates_skipped": train_dupes,
        "eval_duplicates_skipped": eval_dupes,
        "doc_count": len(docs),
        "chunk_count": len(chunks),
        "stage_a_teachers": teacher_names["stage_a"],
        "stage_b_teachers": teacher_names["stage_b"],
        "stage_c_teachers": teacher_names["stage_c"],
        "train_stage_fractions": train_stage_fractions,
        "eval_stage_fractions": eval_stage_fractions,
        "resume_enabled": bool(cfg.output.resume),
        "resume_policy": cfg.output.resume_policy,
        "resume_warning": resume_warning,
        "dry_run": bool(cfg.output.dry_run),
        "dry_run_max_records": int(cfg.output.dry_run_max_records),
        "skipped_records": int(skip_stats.get("skipped_records", 0)),
        "stop_after_stage": cfg.output.stop_after_stage,
        "teacher_startup_checks": startup_checks,
    }
