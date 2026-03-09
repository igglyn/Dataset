"""Configuration schema and TOML loading helpers."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

from distill_factory.teachers.registry import teacher_name_to_backend_type

try:
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # Python 3.10 fallback
    tomllib = None


@dataclass(slots=True)
class DataConfig:
    input_path: str
    file_glob: str
    encoding: str
    chunk_bytes: int
    overlap_bytes: int
    eval_fraction: float
    eval_longdoc_min_bytes: int
    eval_longdoc_fraction: float
    eval_split_strategy: str
    replay_stage_a_fraction: float
    replay_stage_b_fraction: float
    replay_stage_c_fraction: float
    train_replay_stage_a_fraction: float
    train_replay_stage_b_fraction: float
    train_replay_stage_c_fraction: float
    eval_replay_stage_a_fraction: float
    eval_replay_stage_b_fraction: float
    eval_replay_stage_c_fraction: float
    seed: int


@dataclass(slots=True)
class InputConfig:
    preserve_document_boundaries: bool
    normalize_newlines: bool


@dataclass(slots=True)
class OutputConfig:
    output_dir: str
    format: str
    compression: str | None
    max_records_per_shard: int
    shard_prefix: str
    resume: bool
    resume_policy: str
    dry_run: bool
    dry_run_max_records: int
    log_token_lengths: bool
    log_byte_lengths: bool
    stop_after_stage: str | None


@dataclass(slots=True)
class TeacherMixConfig:
    teacher_name: str
    ratio: float


@dataclass(slots=True)
class StageAConfig:
    enabled: bool
    teacher_name: str
    backend_type: str
    mode: str
    top_k: int
    temperature: float
    model_name_or_path: str
    device_map: str
    torch_dtype: str
    max_context: int
    batch_size: int
    tensor_parallel_size: int
    dtype: str
    gpu_memory_utilization: float
    trust_remote_code: bool
    llama_base_url: str
    llama_model_hint: str | None
    llama_request_timeout: float
    extract_hidden_summary: bool
    enable_position_filtering: bool
    entropy_threshold: float | None
    top1_gap_threshold: float | None
    selection_window_radius: int
    selection_mode: str
    minimum_selected_positions_per_record: int | None
    teacher_mixture: list[TeacherMixConfig]

    def record_level_settings(self) -> dict[str, Any]:
        """Stage A settings that are consumed from per-record keys at runtime."""
        return {
            "top_k": int(self.top_k),
            "extract_hidden_summary": bool(self.extract_hidden_summary),
            "enable_position_filtering": bool(self.enable_position_filtering),
            "entropy_threshold": self.entropy_threshold,
            "top1_gap_threshold": self.top1_gap_threshold,
            "selection_window_radius": int(self.selection_window_radius),
            "selection_mode": str(self.selection_mode),
            "minimum_selected_positions_per_record": self.minimum_selected_positions_per_record,
        }


@dataclass(slots=True)
class StageBConfig:
    enabled: bool
    teacher_name: str
    backend_type: str
    mode: str
    top_k: int
    temperature: float
    context_window: int
    stride: int
    window_policy: str
    max_teacher_context: int
    target_region_policy: str
    llama_base_url: str
    llama_model_hint: str | None
    llama_request_timeout: float
    extract_hidden_summary: bool
    enable_position_filtering: bool
    entropy_threshold: float | None
    top1_gap_threshold: float | None
    selection_window_radius: int
    selection_mode: str
    minimum_selected_positions_per_record: int | None
    teacher_mixture: list[TeacherMixConfig]


@dataclass(slots=True)
class StageCConfig:
    enabled: bool
    teacher_name: str
    backend_type: str
    mode: str
    top_k: int
    temperature: float
    task_type: str
    template_name: str
    template_kwargs: dict[str, Any]
    deterministic: bool
    llama_base_url: str
    llama_model_hint: str | None
    llama_request_timeout: float
    extract_hidden_summary: bool
    teacher_mixture: list[TeacherMixConfig]


@dataclass(slots=True)
class PipelineConfig:
    data: DataConfig
    input: InputConfig
    output: OutputConfig
    stage_a: StageAConfig
    stage_b: StageBConfig
    stage_c: StageCConfig


def _parse_value(raw: str) -> Any:
    value = raw.strip()
    if value.startswith('{') and value.endswith('}'):
        inner = value[1:-1].strip()
        if not inner:
            return {}
        out: dict[str, Any] = {}
        for part in inner.split(','):
            if '=' not in part:
                continue
            key, raw_val = [p.strip() for p in part.split('=', 1)]
            key = key.strip('"').strip("'")
            out[key] = _parse_value(raw_val)
        return out
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"none", "null"}:
        return None
    if "." in value:
        return float(value)
    return int(value)


def _parse_minimal_toml(text: str) -> dict[str, dict[str, Any]]:
    """Parse a minimal subset of TOML used by this project."""
    data: dict[str, dict[str, Any]] = {}
    current = ""
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1].strip()
            data.setdefault(current, {})
            continue
        if "=" not in line:
            continue
        key, value = [part.strip() for part in line.split("=", 1)]
        data.setdefault(current, {})[key] = _parse_value(value)
    return data


def _require_section(data: dict[str, Any], section: str) -> dict[str, Any]:
    if section not in data:
        raise ValueError(f"Missing required section: [{section}]")
    section_data = data[section]
    if not isinstance(section_data, dict):
        raise ValueError(f"Section [{section}] must be a table")
    return section_data


def _parse_teacher_mixture(section: dict[str, Any], default_teacher_name: str) -> list[TeacherMixConfig]:
    raw = section.get("teacher_mixture")
    if raw is None:
        return [TeacherMixConfig(teacher_name=default_teacher_name, ratio=1.0)]

    if not isinstance(raw, list):
        raise ValueError("teacher_mixture must be a list of tables with teacher_name and ratio")

    mixture: list[TeacherMixConfig] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("teacher_mixture entries must be tables")
        teacher_name = str(item.get("teacher_name", "")).strip()
        if not teacher_name:
            raise ValueError("teacher_mixture entry missing teacher_name")
        ratio = float(item.get("ratio", 0.0))
        if ratio <= 0.0:
            raise ValueError("teacher_mixture ratio must be > 0")
        mixture.append(TeacherMixConfig(teacher_name=teacher_name, ratio=ratio))

    if not mixture:
        raise ValueError("teacher_mixture must not be empty")
    return mixture


def load_config(path: str | Path) -> PipelineConfig:
    """Load pipeline config from TOML into explicit dataclasses."""
    p = Path(path)
    if tomllib is not None:
        with p.open("rb") as f:
            data: dict[str, Any] = tomllib.load(f)
    else:
        data = _parse_minimal_toml(p.read_text(encoding="utf-8"))

    data_cfg = _require_section(data, "data")
    input_cfg = _require_section(data, "input")
    output_cfg = _require_section(data, "output")
    stage_a_cfg = _require_section(data, "stage_a")
    stage_b_cfg = _require_section(data, "stage_b")
    stage_c_cfg = _require_section(data, "stage_c")

    output_format = str(output_cfg["format"])
    if output_format not in {"jsonl", "parquet"}:
        raise ValueError("[output].format must be 'jsonl' or 'parquet'")


    resume_policy = str(output_cfg.get("resume_policy", "strict"))
    if resume_policy not in {"strict", "best_effort"}:
        raise ValueError("[output].resume_policy must be 'strict' or 'best_effort'")

    dry_run_max_records = int(output_cfg.get("dry_run_max_records", 100))
    if dry_run_max_records <= 0:
        raise ValueError("[output].dry_run_max_records must be > 0")

    log_token_lengths = bool(output_cfg.get("log_token_lengths", False))
    log_byte_lengths = bool(output_cfg.get("log_byte_lengths", False))

    stop_after_stage_raw = output_cfg.get("stop_after_stage")
    stop_after_stage = None if stop_after_stage_raw is None else str(stop_after_stage_raw)
    if stop_after_stage not in {None, "stage_a", "stage_b", "stage_c"}:
        raise ValueError("[output].stop_after_stage must be null, 'stage_a', 'stage_b', or 'stage_c'")

    eval_split_strategy = str(data_cfg.get("eval_split_strategy", "random_docs"))
    if eval_split_strategy not in {"random_docs", "prefer_long_docs"}:
        raise ValueError("[data].eval_split_strategy must be 'random_docs' or 'prefer_long_docs'")

    replay_stage_a_fraction = float(data_cfg.get("replay_stage_a_fraction", 1.0))
    replay_stage_b_fraction = float(data_cfg.get("replay_stage_b_fraction", 1.0))
    replay_stage_c_fraction = float(data_cfg.get("replay_stage_c_fraction", 1.0))

    train_replay_stage_a_fraction = float(data_cfg.get("train_replay_stage_a_fraction", replay_stage_a_fraction))
    train_replay_stage_b_fraction = float(data_cfg.get("train_replay_stage_b_fraction", replay_stage_b_fraction))
    train_replay_stage_c_fraction = float(data_cfg.get("train_replay_stage_c_fraction", replay_stage_c_fraction))

    eval_replay_stage_a_fraction = float(data_cfg.get("eval_replay_stage_a_fraction", replay_stage_a_fraction))
    eval_replay_stage_b_fraction = float(data_cfg.get("eval_replay_stage_b_fraction", replay_stage_b_fraction))
    eval_replay_stage_c_fraction = float(data_cfg.get("eval_replay_stage_c_fraction", replay_stage_c_fraction))

    for name, value in [
        ("replay_stage_a_fraction", replay_stage_a_fraction),
        ("replay_stage_b_fraction", replay_stage_b_fraction),
        ("replay_stage_c_fraction", replay_stage_c_fraction),
        ("train_replay_stage_a_fraction", train_replay_stage_a_fraction),
        ("train_replay_stage_b_fraction", train_replay_stage_b_fraction),
        ("train_replay_stage_c_fraction", train_replay_stage_c_fraction),
        ("eval_replay_stage_a_fraction", eval_replay_stage_a_fraction),
        ("eval_replay_stage_b_fraction", eval_replay_stage_b_fraction),
        ("eval_replay_stage_c_fraction", eval_replay_stage_c_fraction),
    ]:
        if value < 0.0:
            raise ValueError(f"[data].{name} must be >= 0")

    stage_a_mode = str(stage_a_cfg["mode"])
    if stage_a_mode != "topk_logits":
        raise ValueError("[stage_a].mode must be 'topk_logits'")

    stage_b_mode = str(stage_b_cfg["mode"])
    if stage_b_mode not in {"topk_logits", "long_context"}:
        raise ValueError("[stage_b].mode must be 'topk_logits' or 'long_context'")

    stage_b_window_policy = str(stage_b_cfg.get("window_policy", "center_target"))
    if stage_b_window_policy not in {"center_target", "left_biased", "right_biased"}:
        raise ValueError("[stage_b].window_policy must be 'center_target', 'left_biased', or 'right_biased'")

    stage_b_target_region_policy = str(stage_b_cfg.get("target_region_policy", "preserve_full"))
    if stage_b_target_region_policy not in {"preserve_full", "truncate_if_needed"}:
        raise ValueError("[stage_b].target_region_policy must be 'preserve_full' or 'truncate_if_needed'")

    def _validate_position_filtering(stage_key: str, stage_table: dict[str, Any]) -> tuple[bool, float | None, float | None, int, str, int | None]:
        enable_position_filtering = bool(stage_table.get("enable_position_filtering", False))
        selection_mode = str(stage_table.get("selection_mode", "none"))
        if selection_mode not in {"none", "position_mask", "selected_windows"}:
            raise ValueError(
                f"[{stage_key}].selection_mode must be 'none', 'position_mask', or 'selected_windows'"
            )

        entropy_threshold_raw = stage_table.get("entropy_threshold")
        entropy_threshold = None if entropy_threshold_raw is None else float(entropy_threshold_raw)
        if entropy_threshold is not None and (not entropy_threshold >= 0.0):
            raise ValueError(f"[{stage_key}].entropy_threshold must be >= 0 when set")

        top1_gap_threshold_raw = stage_table.get("top1_gap_threshold")
        top1_gap_threshold = None if top1_gap_threshold_raw is None else float(top1_gap_threshold_raw)
        if top1_gap_threshold is not None and (not top1_gap_threshold >= 0.0):
            raise ValueError(f"[{stage_key}].top1_gap_threshold must be >= 0 when set")

        selection_window_radius = int(stage_table.get("selection_window_radius", 0))
        if selection_window_radius < 0:
            raise ValueError(f"[{stage_key}].selection_window_radius must be >= 0")

        minimum_selected_raw = stage_table.get("minimum_selected_positions_per_record")
        minimum_selected_positions_per_record = None if minimum_selected_raw is None else int(minimum_selected_raw)
        if minimum_selected_positions_per_record is not None and minimum_selected_positions_per_record < 0:
            raise ValueError(f"[{stage_key}].minimum_selected_positions_per_record must be >= 0 when set")

        if enable_position_filtering:
            if selection_mode == "none":
                raise ValueError(f"[{stage_key}].selection_mode cannot be 'none' when enable_position_filtering=true")
            if entropy_threshold is None and top1_gap_threshold is None:
                raise ValueError(
                    f"[{stage_key}] position filtering enabled but no thresholds provided; set entropy_threshold and/or top1_gap_threshold"
                )

        return (
            enable_position_filtering,
            entropy_threshold,
            top1_gap_threshold,
            selection_window_radius,
            selection_mode,
            minimum_selected_positions_per_record,
        )

    (
        stage_a_enable_position_filtering,
        stage_a_entropy_threshold,
        stage_a_top1_gap_threshold,
        stage_a_selection_window_radius,
        stage_a_selection_mode,
        stage_a_minimum_selected_positions_per_record,
    ) = _validate_position_filtering("stage_a", stage_a_cfg)

    (
        stage_b_enable_position_filtering,
        stage_b_entropy_threshold,
        stage_b_top1_gap_threshold,
        stage_b_selection_window_radius,
        stage_b_selection_mode,
        stage_b_minimum_selected_positions_per_record,
    ) = _validate_position_filtering("stage_b", stage_b_cfg)

    stage_c_mode = str(stage_c_cfg["mode"])
    if stage_c_mode not in {"structured_outputs", "topk_logits"}:
        raise ValueError("[stage_c].mode must be 'structured_outputs' or 'topk_logits'")

    stage_c_template_name = str(stage_c_cfg.get("template_name", "summarize_chunk"))
    if stage_c_template_name not in {
        "summarize_chunk",
        "answer_question_from_chunk",
        "continue_document",
        "extract_key_points",
    }:
        raise ValueError(
            "[stage_c].template_name must be one of: summarize_chunk, answer_question_from_chunk, continue_document, extract_key_points"
        )

    stage_c_template_kwargs = stage_c_cfg.get("template_kwargs", {})
    if stage_c_template_kwargs is None:
        stage_c_template_kwargs = {}
    if not isinstance(stage_c_template_kwargs, dict):
        raise ValueError("[stage_c].template_kwargs must be a table/dict")

    stage_a_teacher_name = str(stage_a_cfg["teacher_name"])
    stage_b_teacher_name = str(stage_b_cfg["teacher_name"])
    stage_c_teacher_name = str(stage_c_cfg["teacher_name"])

    allowed_backend_types = {"hf", "vllm", "llamacpp_server"}

    def _resolve_backend_type(stage_key: str, stage_table: dict[str, Any], teacher_name: str) -> str:
        configured = str(stage_table.get("backend_type", "")).strip()
        inferred = teacher_name_to_backend_type(teacher_name)
        backend_type = configured or (inferred or "")
        if not backend_type:
            raise ValueError(
                f"[{stage_key}].backend_type is required when teacher_name '{teacher_name}' cannot be mapped automatically"
            )
        if backend_type not in allowed_backend_types:
            raise ValueError(
                f"[{stage_key}].backend_type must be one of: hf, vllm, llamacpp_server"
            )
        return backend_type

    stage_a_backend_type = _resolve_backend_type("stage_a", stage_a_cfg, stage_a_teacher_name)
    stage_b_backend_type = _resolve_backend_type("stage_b", stage_b_cfg, stage_b_teacher_name)
    stage_c_backend_type = _resolve_backend_type("stage_c", stage_c_cfg, stage_c_teacher_name)

    def _validate_llamacpp_backend(stage_key: str, stage_table: dict[str, Any], backend_type: str) -> None:
        if backend_type != "llamacpp_server":
            return
        base_url = str(stage_table.get("llama_base_url", "")).strip()
        if not base_url:
            raise ValueError(f"[{stage_key}].llama_base_url is required when backend_type='llamacpp_server'")
        timeout = float(stage_table.get("llama_request_timeout", 30.0))
        if timeout <= 0:
            raise ValueError(f"[{stage_key}].llama_request_timeout must be > 0")

    _validate_llamacpp_backend("stage_a", stage_a_cfg, stage_a_backend_type)
    _validate_llamacpp_backend("stage_b", stage_b_cfg, stage_b_backend_type)
    _validate_llamacpp_backend("stage_c", stage_c_cfg, stage_c_backend_type)

    os.environ["DISTILL_LLAMACPP_BASE_URL"] = str(stage_a_cfg.get("llama_base_url", "http://127.0.0.1:8080"))
    model_hint = stage_a_cfg.get("llama_model_hint")
    os.environ["DISTILL_LLAMACPP_MODEL_HINT"] = "" if model_hint is None else str(model_hint)
    os.environ["DISTILL_LLAMACPP_REQUEST_TIMEOUT"] = str(float(stage_a_cfg.get("llama_request_timeout", 30.0)))
    os.environ["DISTILL_LLAMACPP_MAX_CONTEXT"] = str(int(stage_a_cfg.get("max_context", 2048)))
    os.environ["DISTILL_LLAMACPP_TOP_K"] = str(int(stage_a_cfg.get("top_k", 5)))
    os.environ["DISTILL_LLAMACPP_TEMPERATURE"] = str(float(stage_a_cfg.get("temperature", 0.0)))

    os.environ["DISTILL_FACTORY_LOG_TOKEN_LENGTHS"] = "1" if log_token_lengths else "0"
    os.environ["DISTILL_FACTORY_LOG_BYTE_LENGTHS"] = "1" if log_byte_lengths else "0"

    return PipelineConfig(
        data=DataConfig(
            input_path=str(data_cfg["input_path"]),
            file_glob=str(data_cfg["file_glob"]),
            encoding=str(data_cfg["encoding"]),
            chunk_bytes=int(data_cfg["chunk_bytes"]),
            overlap_bytes=int(data_cfg["overlap_bytes"]),
            eval_fraction=float(data_cfg["eval_fraction"]),
            eval_longdoc_min_bytes=int(data_cfg.get("eval_longdoc_min_bytes", 0)),
            eval_longdoc_fraction=float(data_cfg.get("eval_longdoc_fraction", 0.0)),
            eval_split_strategy=eval_split_strategy,
            replay_stage_a_fraction=replay_stage_a_fraction,
            replay_stage_b_fraction=replay_stage_b_fraction,
            replay_stage_c_fraction=replay_stage_c_fraction,
            train_replay_stage_a_fraction=train_replay_stage_a_fraction,
            train_replay_stage_b_fraction=train_replay_stage_b_fraction,
            train_replay_stage_c_fraction=train_replay_stage_c_fraction,
            eval_replay_stage_a_fraction=eval_replay_stage_a_fraction,
            eval_replay_stage_b_fraction=eval_replay_stage_b_fraction,
            eval_replay_stage_c_fraction=eval_replay_stage_c_fraction,
            seed=int(data_cfg["seed"]),
        ),
        input=InputConfig(
            preserve_document_boundaries=bool(input_cfg["preserve_document_boundaries"]),
            normalize_newlines=bool(input_cfg["normalize_newlines"]),
        ),
        output=OutputConfig(
            output_dir=str(output_cfg["output_dir"]),
            format=output_format,
            compression=None if output_cfg["compression"] is None else str(output_cfg["compression"]),
            max_records_per_shard=int(output_cfg.get("max_records_per_shard", 0)),
            shard_prefix=str(output_cfg.get("shard_prefix", "shard")),
            resume=bool(output_cfg.get("resume", False)),
            resume_policy=resume_policy,
            dry_run=bool(output_cfg.get("dry_run", False)),
            dry_run_max_records=dry_run_max_records,
            log_token_lengths=log_token_lengths,
            log_byte_lengths=log_byte_lengths,
            stop_after_stage=stop_after_stage,
        ),
        stage_a=StageAConfig(
            enabled=bool(stage_a_cfg["enabled"]),
            teacher_name=stage_a_teacher_name,
            backend_type=stage_a_backend_type,
            mode=stage_a_mode,
            top_k=int(stage_a_cfg["top_k"]),
            temperature=float(stage_a_cfg["temperature"]),
            model_name_or_path=str(stage_a_cfg.get("model_name_or_path", "distilgpt2")),
            device_map=str(stage_a_cfg.get("device_map", "auto")),
            torch_dtype=str(stage_a_cfg.get("torch_dtype", "float16")),
            max_context=int(stage_a_cfg.get("max_context", 2048)),
            batch_size=int(stage_a_cfg.get("batch_size", 1)),
            tensor_parallel_size=int(stage_a_cfg.get("tensor_parallel_size", 1)),
            dtype=str(stage_a_cfg.get("dtype", "auto")),
            gpu_memory_utilization=float(stage_a_cfg.get("gpu_memory_utilization", 0.9)),
            trust_remote_code=bool(stage_a_cfg.get("trust_remote_code", False)),
            llama_base_url=str(stage_a_cfg.get("llama_base_url", "http://127.0.0.1:8080")),
            llama_model_hint=None if stage_a_cfg.get("llama_model_hint") is None else str(stage_a_cfg.get("llama_model_hint")),
            llama_request_timeout=float(stage_a_cfg.get("llama_request_timeout", 30.0)),
            extract_hidden_summary=bool(stage_a_cfg.get("extract_hidden_summary", False)),
            enable_position_filtering=stage_a_enable_position_filtering,
            entropy_threshold=stage_a_entropy_threshold,
            top1_gap_threshold=stage_a_top1_gap_threshold,
            selection_window_radius=stage_a_selection_window_radius,
            selection_mode=stage_a_selection_mode,
            minimum_selected_positions_per_record=stage_a_minimum_selected_positions_per_record,
            teacher_mixture=_parse_teacher_mixture(stage_a_cfg, default_teacher_name=stage_a_teacher_name),
        ),
        stage_b=StageBConfig(
            enabled=bool(stage_b_cfg["enabled"]),
            teacher_name=stage_b_teacher_name,
            backend_type=stage_b_backend_type,
            mode=stage_b_mode,
            top_k=int(stage_b_cfg["top_k"]),
            temperature=float(stage_b_cfg["temperature"]),
            context_window=int(stage_b_cfg["context_window"]),
            stride=int(stage_b_cfg["stride"]),
            window_policy=stage_b_window_policy,
            max_teacher_context=int(stage_b_cfg.get("max_teacher_context", stage_b_cfg.get("context_window", 2048))),
            target_region_policy=stage_b_target_region_policy,
            llama_base_url=str(stage_b_cfg.get("llama_base_url", "http://127.0.0.1:8080")),
            llama_model_hint=None if stage_b_cfg.get("llama_model_hint") is None else str(stage_b_cfg.get("llama_model_hint")),
            llama_request_timeout=float(stage_b_cfg.get("llama_request_timeout", 30.0)),
            extract_hidden_summary=bool(stage_b_cfg.get("extract_hidden_summary", False)),
            enable_position_filtering=stage_b_enable_position_filtering,
            entropy_threshold=stage_b_entropy_threshold,
            top1_gap_threshold=stage_b_top1_gap_threshold,
            selection_window_radius=stage_b_selection_window_radius,
            selection_mode=stage_b_selection_mode,
            minimum_selected_positions_per_record=stage_b_minimum_selected_positions_per_record,
            teacher_mixture=_parse_teacher_mixture(stage_b_cfg, default_teacher_name=stage_b_teacher_name),
        ),
        stage_c=StageCConfig(
            enabled=bool(stage_c_cfg["enabled"]),
            teacher_name=stage_c_teacher_name,
            backend_type=stage_c_backend_type,
            mode=stage_c_mode,
            top_k=int(stage_c_cfg["top_k"]),
            temperature=float(stage_c_cfg["temperature"]),
            task_type=str(stage_c_cfg.get("task_type", "refinement")),
            template_name=stage_c_template_name,
            template_kwargs=dict(stage_c_template_kwargs),
            deterministic=bool(stage_c_cfg.get("deterministic", True)),
            llama_base_url=str(stage_c_cfg.get("llama_base_url", "http://127.0.0.1:8080")),
            llama_model_hint=None if stage_c_cfg.get("llama_model_hint") is None else str(stage_c_cfg.get("llama_model_hint")),
            llama_request_timeout=float(stage_c_cfg.get("llama_request_timeout", 30.0)),
            extract_hidden_summary=bool(stage_c_cfg.get("extract_hidden_summary", False)),
            teacher_mixture=_parse_teacher_mixture(stage_c_cfg, default_teacher_name=stage_c_teacher_name),
        ),
    )


# Backward-compatible alias from the initial scaffold.
load_config_toml = load_config

from distill_factory.corpus.schema import (
    CorpusMixtureConfig,
    MixtureBuildConfig,
    MixtureDatasetConfig,
    MixtureGroupConfig,
    SourceDatasetCacheConfig,
    SourceExtractionConfig,
    load_corpus_mixture_config,
)
