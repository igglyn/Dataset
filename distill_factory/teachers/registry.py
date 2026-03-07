"""Teacher registry."""

from __future__ import annotations

import os
from typing import Callable

from .base import DummyTeacher, Teacher
from .hf_causal_lm import HFCausalLMTeacher
from .vllm_causal_lm import VLLMCausalLMTeacher


TeacherFactory = Callable[[], Teacher]
_TEACHER_REGISTRY: dict[str, TeacherFactory] = {}


_TEACHER_CAPABILITIES: dict[str, dict[str, bool]] = {
    "dummy": {
        "supports_topk": True,
        "supports_structured": True,
        "supports_long_context": False,
        "supports_hidden_summary": False,
    },
    "bulk_grounding_teacher": {
        "supports_topk": True,
        "supports_structured": True,
        "supports_long_context": False,
        "supports_hidden_summary": False,
    },
    "long_context_structure_teacher": {
        "supports_topk": True,
        "supports_structured": True,
        "supports_long_context": False,
        "supports_hidden_summary": False,
    },
    "refinement_teacher": {
        "supports_topk": True,
        "supports_structured": True,
        "supports_long_context": False,
        "supports_hidden_summary": False,
    },
    "hf_causal_lm": {
        "supports_topk": True,
        "supports_structured": True,
        "supports_long_context": True,
        "supports_hidden_summary": True,
    },
    "vllm_causal_lm": {
        "supports_topk": True,
        "supports_structured": False,
        "supports_long_context": True,
        "supports_hidden_summary": False,
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
) -> None:
    """Validate teacher capabilities for a stage/mode and fail early with clear errors."""
    missing: list[str] = []
    if require_topk and not _teacher_capability(teacher, teacher_name, "supports_topk"):
        missing.append("supports_topk")
    if require_structured and not _teacher_capability(teacher, teacher_name, "supports_structured"):
        missing.append("supports_structured")
    if require_hidden_summary and not _teacher_capability(teacher, teacher_name, "supports_hidden_summary"):
        missing.append("supports_hidden_summary")
    if require_long_context and not _teacher_capability(teacher, teacher_name, "supports_long_context"):
        missing.append("supports_long_context")

    if missing:
        mode_suffix = f" (mode={mode})" if mode else ""
        raise ValueError(
            f"Teacher '{teacher_name}' ({type(teacher).__name__}) cannot run {stage_name}{mode_suffix}; "
            f"missing required capabilities: {', '.join(missing)}."
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
    """
    return HFCausalLMTeacher(
        model_name_or_path=os.getenv("DISTILL_HF_MODEL_NAME_OR_PATH", "distilgpt2"),
        device_map=os.getenv("DISTILL_HF_DEVICE_MAP", "auto"),
        torch_dtype=os.getenv("DISTILL_HF_TORCH_DTYPE", "float16"),
        max_context=int(os.getenv("DISTILL_HF_MAX_CONTEXT", "2048")),
        batch_size=int(os.getenv("DISTILL_HF_BATCH_SIZE", "1")),
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
