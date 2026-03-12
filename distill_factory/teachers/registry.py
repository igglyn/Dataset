"""Teacher registry."""

from __future__ import annotations

import os
from typing import Callable, Literal

from .base import DummyTeacher, Teacher
from .hf_causal_lm import HFCausalLMTeacher
from .llamacpp_server import LlamaCppServerTeacher
from .vllm_causal_lm import VLLMCausalLMTeacher



RuntimeBackendType = Literal["hf", "vllm", "llamacpp_server"]


def teacher_name_to_backend_type(teacher_name: str) -> RuntimeBackendType | None:
    """Map teacher identifiers to runtime backend types when known."""
    mapping: dict[str, RuntimeBackendType] = {
        "dummy": "hf",
        "bulk_grounding_teacher": "hf",
        "long_context_structure_teacher": "hf",
        "refinement_teacher": "hf",
        "hf_causal_lm": "hf",
        "vllm_causal_lm": "vllm",
        "llamacpp_server": "llamacpp_server",
    }
    return mapping.get(teacher_name)


TeacherFactory = Callable[[], Teacher]
_TEACHER_REGISTRY: dict[str, TeacherFactory] = {}


_TEACHER_CAPABILITIES: dict[str, dict[str, object]] = {
    "dummy": {
        "backend_type": "hf",
        "supports_topk": True,
        "supports_structured": True,
        "supports_long_context": False,
        "supports_hidden_summary": False,
        "supports_per_token_entropy": False,
        "supports_per_token_top1_gap": False,
    },
    "bulk_grounding_teacher": {
        "backend_type": "hf",
        "supports_topk": True,
        "supports_structured": True,
        "supports_long_context": False,
        "supports_hidden_summary": False,
        "supports_per_token_entropy": False,
        "supports_per_token_top1_gap": False,
    },
    "long_context_structure_teacher": {
        "backend_type": "hf",
        "supports_topk": True,
        "supports_structured": True,
        "supports_long_context": False,
        "supports_hidden_summary": False,
        "supports_per_token_entropy": False,
        "supports_per_token_top1_gap": False,
    },
    "refinement_teacher": {
        "backend_type": "hf",
        "supports_topk": True,
        "supports_structured": True,
        "supports_long_context": False,
        "supports_hidden_summary": False,
        "supports_per_token_entropy": False,
        "supports_per_token_top1_gap": False,
    },
    "hf_causal_lm": {
        "backend_type": "hf",
        "supports_topk": True,
        "supports_structured": True,
        "supports_long_context": True,
        "supports_hidden_summary": True,
        "supports_per_token_entropy": True,
        "supports_per_token_top1_gap": True,
    },
    "vllm_causal_lm": {
        "backend_type": "vllm",
        "supports_topk": True,
        "supports_structured": False,
        "supports_long_context": True,
        "supports_hidden_summary": False,
        "supports_per_token_entropy": True,
        "supports_per_token_top1_gap": True,
    },
    "llamacpp_server": {
        "backend_type": "llamacpp_server",
        "supports_topk": True,
        "supports_structured": False,
        "supports_long_context": True,
        "supports_hidden_summary": False,
        "supports_per_token_entropy": True,
        "supports_per_token_top1_gap": True,
    },
}


def _teacher_capability(teacher: Teacher, teacher_name: str, capability_name: str) -> bool:
    profile = _TEACHER_CAPABILITIES.get(teacher_name, {})
    if capability_name in profile:
        return bool(profile[capability_name])
    attr = getattr(teacher, capability_name, None)
    if callable(attr):
        return bool(attr())
    return False


def validate_teacher_capabilities(
    teacher: Teacher,
    teacher_name: str,
    *,
    stage_name: str,
    mode: str | None = None,
    require_topk: bool = False,
    require_structured: bool = False,
    require_hidden_summary: bool = False,
    require_long_context: bool = False,
    require_per_token_entropy: bool = False,
    require_per_token_top1_gap: bool = False,
) -> None:
    """Validate teacher capabilities for a stage/mode and fail early with clear errors.

    Note: per-token capability flags describe whether a teacher can emit per-token
    signals, not whether a stage currently implements selection-aware export behavior.
    """
    missing: list[str] = []
    if require_topk and not _teacher_capability(teacher, teacher_name, "supports_topk"):
        missing.append("supports_topk")
    if require_structured and not _teacher_capability(teacher, teacher_name, "supports_structured"):
        missing.append("supports_structured")
    if require_hidden_summary and not _teacher_capability(teacher, teacher_name, "supports_hidden_summary"):
        missing.append("supports_hidden_summary")
    if require_long_context and not _teacher_capability(teacher, teacher_name, "supports_long_context"):
        missing.append("supports_long_context")
    if require_per_token_entropy and not _teacher_capability(teacher, teacher_name, "supports_per_token_entropy"):
        missing.append("supports_per_token_entropy")
    if require_per_token_top1_gap and not _teacher_capability(teacher, teacher_name, "supports_per_token_top1_gap"):
        missing.append("supports_per_token_top1_gap")

    if missing:
        mode_suffix = f" (mode={mode})" if mode else ""
        raise ValueError(
            f"Teacher '{teacher_name}' ({type(teacher).__name__}) cannot run {stage_name}{mode_suffix}; "
            f"missing required capabilities: {', '.join(missing)}."
        )




def validate_teacher_backend_compatibility(teacher_name: str, backend_type: str) -> None:
    """Validate explicit backend selection against known built-in teachers."""
    inferred = teacher_name_to_backend_type(teacher_name)
    if inferred is not None and inferred != backend_type:
        raise ValueError(
            f"Teacher '{teacher_name}' is tied to backend_type '{inferred}', got '{backend_type}'."
        )

def register_teacher(name: str) -> Callable[[TeacherFactory], TeacherFactory]:
    """Decorator to register a teacher factory by name."""

    def decorator(factory: TeacherFactory) -> TeacherFactory:
        _TEACHER_REGISTRY[name] = factory
        return factory

    return decorator


def get_teacher(name: str) -> Teacher:
    """Instantiate a registered teacher by name."""
    if name not in _TEACHER_REGISTRY:
        raise KeyError(f"Unknown teacher: {name}")
    return _TEACHER_REGISTRY[name]()


@register_teacher("dummy")
def _build_dummy_teacher() -> Teacher:
    return DummyTeacher(name="dummy")


@register_teacher("bulk_grounding_teacher")
def _build_stage_a_teacher() -> Teacher:
    return DummyTeacher(name="bulk_grounding_teacher")


@register_teacher("long_context_structure_teacher")
def _build_stage_b_teacher() -> Teacher:
    return DummyTeacher(name="long_context_structure_teacher")


@register_teacher("refinement_teacher")
def _build_stage_c_teacher() -> Teacher:
    return DummyTeacher(name="refinement_teacher")


@register_teacher("hf_causal_lm")
def _build_hf_causal_lm_teacher() -> Teacher:
    """HF backend factory configured via env vars for minimal integration.

    Env vars:
    - DISTILL_HF_MODEL_NAME_OR_PATH
    - DISTILL_HF_DEVICE_MAP
    - DISTILL_HF_TORCH_DTYPE
    - DISTILL_HF_MAX_CONTEXT
    - DISTILL_HF_BATCH_SIZE
    - DISTILL_HF_PAD_TOKEN_ID (optional int)
    - DISTILL_HF_OFFLOAD_LAYERS (optional int)
    """
    return HFCausalLMTeacher(
        model_name_or_path=os.getenv("DISTILL_HF_MODEL_NAME_OR_PATH", "distilgpt2"),
        device_map=os.getenv("DISTILL_HF_DEVICE_MAP", "auto"),
        torch_dtype=os.getenv("DISTILL_HF_TORCH_DTYPE", "float16"),
        max_context=int(os.getenv("DISTILL_HF_MAX_CONTEXT", "2048")),
        batch_size=int(os.getenv("DISTILL_HF_BATCH_SIZE", "1")),
        hf_pad_token_id=(
            None
            if os.getenv("DISTILL_HF_PAD_TOKEN_ID", "").strip() == ""
            else int(os.getenv("DISTILL_HF_PAD_TOKEN_ID", "0"))
        ),
        hf_offload_layers=(
            None
            if os.getenv("DISTILL_HF_OFFLOAD_LAYERS", "").strip() == ""
            else int(os.getenv("DISTILL_HF_OFFLOAD_LAYERS", "0"))
        ),
    )


@register_teacher("vllm_causal_lm")
def _build_vllm_causal_lm_teacher() -> Teacher:
    """vLLM backend factory configured via env vars for minimal integration."""
    return VLLMCausalLMTeacher(
        model_name_or_path=os.getenv("DISTILL_VLLM_MODEL_NAME_OR_PATH", "Qwen/Qwen2.5-0.5B"),
        tensor_parallel_size=int(os.getenv("DISTILL_VLLM_TENSOR_PARALLEL_SIZE", "1")),
        dtype=os.getenv("DISTILL_VLLM_DTYPE", "auto"),
        max_context=int(os.getenv("DISTILL_VLLM_MAX_CONTEXT", "2048")),
        batch_size=int(os.getenv("DISTILL_VLLM_BATCH_SIZE", "1")),
        gpu_memory_utilization=float(os.getenv("DISTILL_VLLM_GPU_MEMORY_UTILIZATION", "0.9")),
        trust_remote_code=os.getenv("DISTILL_VLLM_TRUST_REMOTE_CODE", "false").lower() == "true",
    )


@register_teacher("llamacpp_server")
def _build_llamacpp_server_teacher() -> Teacher:
    """llama.cpp server backend factory configured via env vars."""
    return LlamaCppServerTeacher(
        base_url=os.getenv("DISTILL_LLAMACPP_BASE_URL", "http://127.0.0.1:8080"),
        model_hint=os.getenv("DISTILL_LLAMACPP_MODEL_HINT") or None,
        request_timeout=float(os.getenv("DISTILL_LLAMACPP_REQUEST_TIMEOUT", "30.0")),
        max_context=int(os.getenv("DISTILL_LLAMACPP_MAX_CONTEXT", "2048")),
        default_top_k=int(os.getenv("DISTILL_LLAMACPP_TOP_K", "5")),
        default_temperature=float(os.getenv("DISTILL_LLAMACPP_TEMPERATURE", "0.0")),
    )
