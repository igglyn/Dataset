"""Manifest helpers for reusable source extraction caches."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from distill_factory.corpus.schema import SourceDatasetCacheConfig

MANIFEST_FILENAME = "manifest.json"


def source_config_payload(source: SourceDatasetCacheConfig) -> dict[str, Any]:
    return {
        "source_name": source.source_name,
        "source_type": source.source_type,
        "hf_dataset": source.hf_dataset,
        "hf_config": source.hf_config,
        "split_mapping": dict(sorted(source.split_mapping.items())),
        "text_field": source.text_field,
        "group_size": source.group_size,
        "max_docs_per_split": source.max_docs_per_split,
        "min_bytes": source.min_bytes,
        "max_bytes": source.max_bytes,
    }


def config_fingerprint(source: SourceDatasetCacheConfig) -> str:
    payload = source_config_payload(source)
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_manifest(source_dir: Path) -> dict[str, Any] | None:
    path = source_dir / MANIFEST_FILENAME
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(source_dir: Path, manifest: dict[str, Any]) -> None:
    path = source_dir / MANIFEST_FILENAME
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def manifest_matches_source(manifest: dict[str, Any], source: SourceDatasetCacheConfig) -> bool:
    return str(manifest.get("config_fingerprint", "")) == config_fingerprint(source)
