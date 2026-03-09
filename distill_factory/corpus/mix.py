"""Build deterministic mixed text corpora from previously cached source datasets."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
from typing import Any

from distill_factory.corpus.schema import CorpusMixtureConfig

CANONICAL_SPLITS = ("train", "eval", "validation")
VALID_DEPLETION_POLICIES = ("rebalance", "strict", "record_only")


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_group_percentages(config: CorpusMixtureConfig) -> None:
    total = sum(group.percentage for group in config.mixture_build.groups)
    if abs(total - 100.0) > 1e-9:
        raise ValueError(f"mixture group percentages must sum to 100, got {total}")


def _allocate_group_counts(target_documents: int, group_percentages: dict[str, float]) -> dict[str, int]:
    raw = {
        name: target_documents * (pct / 100.0)
        for name, pct in group_percentages.items()
    }
    base = {name: int(value) for name, value in raw.items()}
    remainder = target_documents - sum(base.values())

    if remainder > 0:
        ranked = sorted(
            group_percentages.keys(),
            key=lambda name: (raw[name] - base[name], name),
            reverse=True,
        )
        for name in ranked[:remainder]:
            base[name] += 1
    return base


def _list_cached_docs(cache_root: Path, source_name: str, split: str) -> list[Path]:
    split_dir = cache_root / source_name / split
    if not split_dir.is_dir():
        raise ValueError(
            f"Missing cached split directory for source '{source_name}' split '{split}': {split_dir}"
        )
    docs = sorted(split_dir.glob("doc_*.txt"))
    if not docs:
        raise ValueError(f"No cached documents found for source '{source_name}' split '{split}'")
    return docs




def _doc_byte_length(path: Path) -> int:
    return len(path.read_bytes())


def _passes_mixture_byte_filters(path: Path, min_bytes: int | None, max_bytes: int | None) -> tuple[bool, int]:
    size = _doc_byte_length(path)
    if min_bytes is not None and size < min_bytes:
        return False, size
    if max_bytes is not None and size > max_bytes:
        return False, size
    return True, size

def _load_source_cache_refs(cache_root: Path, config: CorpusMixtureConfig) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for source in config.source_extraction.datasets:
        source_dir = cache_root / source.source_name
        manifest_path = source_dir / "manifest.json"
        ref: dict[str, Any] = {
            "source_name": source.source_name,
            "source_dir": str(source_dir),
            "manifest_path": str(manifest_path),
            "manifest_exists": manifest_path.is_file(),
        }
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                ref["config_fingerprint"] = manifest.get("config_fingerprint")
                ref["extraction_timestamp"] = manifest.get("extraction_timestamp")
            except Exception:
                ref["manifest_parse_error"] = "failed_to_parse_manifest"
        refs.append(ref)
    return refs


def build_corpus_mixture_from_cache(
    config: CorpusMixtureConfig,
    *,
    mixture_name: str,
    cache_root: str | Path,
    output_root: str | Path,
    dry_run: bool = False,
) -> Path:
    """Build a mixed corpus from cached source docs only (no source re-extraction)."""
    _validate_group_percentages(config)

    if config.mixture_build.target_documents <= 0:
        raise ValueError("target_documents must be > 0")

    if not mixture_name.strip():
        raise ValueError("mixture_name must be non-empty")

    cache_root_path = Path(cache_root)
    mixture_dir = Path(output_root) / mixture_name

    if dry_run:
        return mixture_dir

    if mixture_dir.exists():
        raise ValueError(
            f"Output mixture directory already exists: {mixture_dir}. "
            "Use a different mixture_name or remove existing output first."
        )

    group_percentages = {group.group_name: float(group.percentage) for group in config.mixture_build.groups}
    depletion_policy = str(config.mixture_build.depletion_policy)
    if depletion_policy not in VALID_DEPLETION_POLICIES:
        raise ValueError(f"Unsupported depletion_policy: {depletion_policy}")

    split_summaries: list[dict[str, Any]] = []
    split_offset = {"train": 0, "eval": 1, "validation": 2}
    warnings: list[str] = []

    mixture_min_bytes = config.mixture_build.min_bytes
    mixture_max_bytes = config.mixture_build.max_bytes

    realized_group_counts_total: dict[str, int] = defaultdict(int)
    realized_source_counts_total: dict[str, int] = defaultdict(int)
    realized_split_counts: dict[str, int] = {}

    for split in CANONICAL_SPLITS:
        (mixture_dir / split).mkdir(parents=True, exist_ok=True)
        target = int(config.mixture_build.target_documents)
        requested_by_group = _allocate_group_counts(target, group_percentages)

        sampled_rows: list[dict[str, Any]] = []
        rng = random.Random(int(config.mixture_build.random_seed) + split_offset[split])

        realized_by_group_split: dict[str, int] = defaultdict(int)
        requested_by_group_split: dict[str, int] = dict(requested_by_group)

        remaining_by_group: dict[str, list[dict[str, Any]]] = {}
        filtered_below_min_bytes = 0
        filtered_above_max_bytes = 0

        for group in config.mixture_build.groups:
            group_name = group.group_name
            requested = requested_by_group[group_name]
            if requested == 0:
                remaining_by_group[group_name] = []
                continue

            pool: list[dict[str, Any]] = []
            for dataset in group.datasets:
                docs = _list_cached_docs(cache_root_path, dataset.source_name, split)
                for doc_path in docs:
                    keep, byte_len = _passes_mixture_byte_filters(
                        doc_path,
                        min_bytes=mixture_min_bytes,
                        max_bytes=mixture_max_bytes,
                    )
                    if not keep:
                        if mixture_min_bytes is not None and byte_len < mixture_min_bytes:
                            filtered_below_min_bytes += 1
                        elif mixture_max_bytes is not None and byte_len > mixture_max_bytes:
                            filtered_above_max_bytes += 1
                        continue
                    pool.append(
                        {
                            "group_name": group_name,
                            "source_name": dataset.source_name,
                            "source_split": split,
                            "source_doc_path": str(doc_path),
                            "source_doc_id": doc_path.stem,
                            "byte_length": byte_len,
                        }
                    )

            rng.shuffle(pool)
            selected = pool[:requested]
            remaining = pool[requested:]
            remaining_by_group[group_name] = remaining

            if len(selected) < requested:
                if depletion_policy == "strict":
                    raise ValueError(
                        f"Split '{split}' group '{group_name}' requested {requested} docs but only {len(pool)} "
                        "were available under strict depletion_policy."
                    )
                if depletion_policy == "rebalance":
                    warnings.append(
                        f"Split '{split}' group '{group_name}' requested {requested} docs but only {len(pool)} "
                        "were available; attempting deterministic rebalance from other groups."
                    )
                else:
                    warnings.append(
                        f"Split '{split}' group '{group_name}' requested {requested} docs but only {len(pool)} "
                        "were available; record_only policy will keep this shortfall."
                    )

            for row in selected:
                sampled_rows.append(row)
                realized_by_group_split[group_name] += 1
                realized_group_counts_total[group_name] += 1
                realized_source_counts_total[row["source_name"]] += 1

        shortfall = target - len(sampled_rows)
        if depletion_policy == "rebalance":
            while shortfall > 0:
                candidates = [name for name in sorted(remaining_by_group) if remaining_by_group[name]]
                if not candidates:
                    break
                rng.shuffle(candidates)
                picked_group = candidates[0]
                row = remaining_by_group[picked_group].pop(0)
                sampled_rows.append(row)
                realized_by_group_split[picked_group] += 1
                realized_group_counts_total[picked_group] += 1
                realized_source_counts_total[row["source_name"]] += 1
                shortfall -= 1

        if filtered_below_min_bytes > 0 or filtered_above_max_bytes > 0:
            warnings.append(
                f"Split '{split}' mixture byte filtering excluded docs: below_min={filtered_below_min_bytes}, "
                f"above_max={filtered_above_max_bytes}."
            )

        if len(sampled_rows) < target:
            if depletion_policy == "strict":
                raise ValueError(
                    f"Split '{split}' realized {len(sampled_rows)} / {target} under strict depletion_policy"
                )
            warnings.append(
                f"Split '{split}' realized {len(sampled_rows)} / {target} documents under depletion_policy="
                f"{depletion_policy}."
            )

        rng.shuffle(sampled_rows)

        for out_idx, row in enumerate(sampled_rows, start=1):
            out_stem = f"doc_{out_idx:08d}"
            source_doc_path = Path(row["source_doc_path"])
            text = source_doc_path.read_text(encoding="utf-8")

            (mixture_dir / split / f"{out_stem}.txt").write_text(text, encoding="utf-8")
            sidecar = {
                "mixture_name": mixture_name,
                "group_name": row["group_name"],
                "source_name": row["source_name"],
                "source_split": row["source_split"],
                "source_doc_path": row["source_doc_path"],
                "source_doc_id": row["source_doc_id"],
                "source_doc_bytes": row.get("byte_length"),
                "output_doc_id": out_stem,
                "build_timestamp": _utc_timestamp(),
            }
            (mixture_dir / split / f"{out_stem}.meta.json").write_text(
                json.dumps(sidecar, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        realized_split_counts[split] = len(sampled_rows)
        split_summaries.append(
            {
                "split": split,
                "target_documents": target,
                "requested_by_group": requested_by_group_split,
                "realized_by_group": dict(realized_by_group_split),
                "output_documents": len(sampled_rows),
                "deviation_from_target": len(sampled_rows) != target,
                "depletion_policy": depletion_policy,
                "mixture_min_bytes": mixture_min_bytes,
                "mixture_max_bytes": mixture_max_bytes,
                "filtered_below_min_bytes": filtered_below_min_bytes,
                "filtered_above_max_bytes": filtered_above_max_bytes,
            }
        )

    target_total = int(config.mixture_build.target_documents) * len(CANONICAL_SPLITS)
    realized_total = sum(realized_split_counts.values())

    manifest = {
        "mixture_name": mixture_name,
        "created_at": _utc_timestamp(),
        "cache_root": str(cache_root_path),
        "output_root": str(mixture_dir),
        "target_documents": int(config.mixture_build.target_documents),
        "target_documents_per_split": int(config.mixture_build.target_documents),
        "random_seed": int(config.mixture_build.random_seed),
        "requested_group_percentages": group_percentages,
        "mixture_min_bytes": mixture_min_bytes,
        "mixture_max_bytes": mixture_max_bytes,
        "realized_document_counts_by_group": dict(realized_group_counts_total),
        "realized_document_counts_by_source_dataset": dict(realized_source_counts_total),
        "realized_split_counts": realized_split_counts,
        "source_cache_references": _load_source_cache_refs(cache_root_path, config),
        "requested_total_documents": target_total,
        "realized_total_documents": realized_total,
        "composition_deviation": realized_total != target_total,
        "depletion_policy": depletion_policy,
        "warnings": warnings,
        "splits": split_summaries,
    }
    (mixture_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return mixture_dir
