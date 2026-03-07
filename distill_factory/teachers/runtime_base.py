"""Runtime/backend interfaces used by teacher implementations.

This module separates *teacher semantics* (what the pipeline asks for) from
*runtime transport/execution* (how model inference is performed).

Teacher classes may delegate execution to a runtime that implements
``TeacherRuntime``. Local Python runtimes (HF/vLLM) and external transports
(e.g. llama.cpp server) can share this shape.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RuntimeCapabilities:
    """Capability summary reported by a runtime backend."""

    backend_type: str
    supports_topk: bool
    supports_structured: bool
    supports_tokenizer_diagnostics: bool


class TeacherRuntime(ABC):
    """Execution backend interface for teacher models.

    Notes:
    - ``infer_topk`` is required by the distillation path.
    - ``infer_structured`` and ``tokenizer_diagnostics`` are optional and should
      raise ``NotImplementedError`` when unsupported.
    - ``startup_self_check`` is a fast preflight hook used before long runs.
    """

    @abstractmethod
    def startup_self_check(self) -> None:
        """Run a fast backend validation and fail early on misconfiguration."""

    @abstractmethod
    def infer_topk(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return top-k token/logprob outputs for each input record."""

    def infer_structured(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return one structured generation output per record (optional)."""
        raise NotImplementedError("This runtime backend does not support structured generation.")

    def tokenizer_diagnostics(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return tokenizer-only diagnostics for records (optional)."""
        raise NotImplementedError("This runtime backend does not support tokenizer diagnostics.")

    @abstractmethod
    def capabilities(self) -> RuntimeCapabilities:
        """Return backend capability flags for explicit pipeline checks."""

    @abstractmethod
    def close(self) -> None:
        """Release runtime resources (engines, sessions, sockets, etc.)."""
