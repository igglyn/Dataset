"""Helpers for explicit, human-readable pipeline resume state."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


RESUME_STATE_NAME = "resume_state.json"
RESUME_SCHEMA_VERSION = "1"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _critical_config_payload(cfg: Any) -> dict[str, Any]:
    return {
        "data": asdict(cfg.data),
        "input": asdict(cfg.input),
        "stage_a": asdict(cfg.stage_a),
        "stage_b": asdict(cfg.stage_b),
        "stage_c": asdict(cfg.stage_c),
        "output_format": cfg.output.format,
    }


def _full_config_payload(cfg: Any) -> dict[str, Any]:
    return asdict(cfg)


def config_fingerprints(cfg: Any) -> dict[str, str]:
    """Return stable critical/full config fingerprints for resume validation."""
    return {
        "critical": _sha256_text(_json_dumps(_critical_config_payload(cfg))),
        "full": _sha256_text(_json_dumps(_full_config_payload(cfg))),
    }


def load_resume_state(output_dir: str | Path) -> dict[str, Any] | None:
    path = Path(output_dir) / RESUME_STATE_NAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_resume_state(output_dir: str | Path, state: dict[str, Any]) -> Path:
    path = Path(output_dir) / RESUME_STATE_NAME
    state = dict(state)
    state["last_update_timestamp"] = _now_iso()
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    return path


def build_initial_resume_state(cfg: Any, config_path: str, teacher_names: dict[str, list[str]]) -> dict[str, Any]:
    fps = config_fingerprints(cfg)
    return {
        "schema_version": RESUME_SCHEMA_VERSION,
        "config_path": str(config_path),
        "config_fingerprint": fps["full"],
        "critical_config_fingerprint": fps["critical"],
        "completed_stages": [],
        "split_progress": {
            "train": {"record_count": 0, "shard_ids": []},
            "eval": {"record_count": 0, "shard_ids": []},
        },
        "teacher_names": teacher_names,
        "last_update_timestamp": _now_iso(),
    }


def validate_resume_state(state: dict[str, Any], cfg: Any, resume_policy: str) -> tuple[bool, str | None]:
    """Validate config compatibility under strict or best_effort policy."""
    fps = config_fingerprints(cfg)
    saved_full = str(state.get("config_fingerprint", ""))
    saved_critical = str(state.get("critical_config_fingerprint", ""))

    if resume_policy == "strict":
        if saved_full and saved_full != fps["full"]:
            return False, "resume strict mode: config fingerprint differs from prior run"
        return True, None

    if saved_critical and saved_critical != fps["critical"]:
        return False, "resume best_effort mode: critical config fields differ from prior run"

    if saved_full and saved_full != fps["full"]:
        return True, "resume best_effort: non-critical config fields changed; continuing"

    return True, None
