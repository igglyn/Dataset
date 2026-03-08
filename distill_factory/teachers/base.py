"""Teacher interfaces and a dummy implementation for smoke testing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TypedDict

from .runtime_base import RuntimeCapabilities, TeacherRuntime


class LongContextInput(TypedDict, total=False):
    """Common long-context payload fields shared by teacher backends."""

    teacher_input_bytes: bytes
    teacher_input_text: str
    target_start_offset: int
    target_end_offset: int
    long_context_truncation: dict[str, Any]


class Teacher(ABC):
    """Abstract teacher interface shared across pipeline stages.

    Optional hidden summary policy:
    - `hidden_summary` is a compact per-record vector summary of internal teacher states.
    - Backends may provide it (e.g., pooled hidden states) via `infer_topk` outputs.

    Capability methods are used by stage runners for explicit fail-fast validation.
    """

    @abstractmethod
    def prepare(self) -> None:
        """Prepare resources needed before inference."""

    @abstractmethod
    def infer_topk(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return top-k outputs per record: top_k_ids, top_k_logprobs, entropy."""

    @abstractmethod
    def infer_structured(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return one structured output dict per record.

        Expected structured schema:
        - task_type: str
        - prompt_text: str
        - completion_text: str
        - teacher_metadata: dict | None
        """


    def supports_hidden_summary(self) -> bool:
        """Whether this teacher backend can emit `hidden_summary`."""
        return False


    def supports_topk(self) -> bool:
        """Whether this teacher backend supports top-k inference."""
        return False

    def supports_structured(self) -> bool:
        """Whether this teacher backend supports structured outputs."""
        return False

    def supports_long_context(self) -> bool:
        """Whether this teacher backend supports stage-B long-context handling."""
        return False

    def supports_tokenizer_diagnostics(self) -> bool:
        """Whether this teacher backend supports tokenizer-only diagnostics."""
        return False


    def supports_per_token_entropy(self) -> bool:
        """Whether this teacher backend can emit per-token entropy signals."""
        return False

    def supports_per_token_top1_gap(self) -> bool:
        """Whether this teacher backend can emit per-token top1-gap signals."""
        return False

    @abstractmethod
    def close(self) -> None:
        """Release any resources held by this teacher."""


class RuntimeBackedTeacher(Teacher):
    """Teacher helper that delegates execution to a runtime backend.

    This keeps teacher-facing semantics stable while allowing execution to move
    across local runtimes (HF/vLLM) or remote transports.
    """

    def __init__(self, runtime: TeacherRuntime) -> None:
        self.runtime = runtime

    def runtime_capabilities(self) -> RuntimeCapabilities:
        """Expose backend capability flags for diagnostics and validation."""
        return self.runtime.capabilities()

    def prepare(self) -> None:
        self.runtime.startup_self_check()

    def infer_topk(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return self.runtime.infer_topk(records)

    def infer_structured(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return self.runtime.infer_structured(records)

    def supports_topk(self) -> bool:
        return self.runtime.capabilities().supports_topk

    def supports_structured(self) -> bool:
        return self.runtime.capabilities().supports_structured

    def close(self) -> None:
        self.runtime.close()


class DummyTeacher(Teacher):
    """Deterministic fake teacher for pipeline smoke tests."""

    def __init__(self, name: str = "dummy_teacher", top_k: int = 4) -> None:
        self.name = name
        self.top_k = top_k
        self._prepared = False

    def supports_topk(self) -> bool:
        return True

    def supports_structured(self) -> bool:
        return True

    def supports_long_context(self) -> bool:
        return False

    def supports_hidden_summary(self) -> bool:
        return False

    def prepare(self) -> None:
        self._prepared = True

    def infer_topk(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self._prepared:
            raise RuntimeError("Teacher must be prepared before inference.")
        if any(bool(r.get("extract_hidden_summary", False)) for r in records):
            raise RuntimeError("DummyTeacher does not support hidden_summary; disable extract_hidden_summary or use a backend that supports it.")

        outputs: list[dict[str, Any]] = []
        ids = list(range(self.top_k))
        logprobs = [-(i + 1) * 0.1 for i in range(self.top_k)]
        entropy = float(self.top_k) * 0.1

        for _ in records:
            outputs.append(
                {
                    "top_k_ids": ids,
                    "top_k_logprobs": logprobs,
                    "entropy": entropy,
                }
            )
        return outputs

    def infer_structured(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self._prepared:
            raise RuntimeError("Teacher must be prepared before inference.")

        outputs: list[dict[str, Any]] = []
        for idx, record in enumerate(records):
            prompt_text = str(record.get("prompt_text", ""))
            outputs.append(
                {
                    "task_type": str(record.get("task_type", "refinement")),
                    "prompt_text": prompt_text,
                    "completion_text": f"dummy_completion_{idx}",
                    "teacher_metadata": {"teacher": self.name},
                }
            )
        return outputs

    def close(self) -> None:
        self._prepared = False
