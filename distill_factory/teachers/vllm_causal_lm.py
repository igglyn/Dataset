"""vLLM causal LM teacher backend for top-k distillation outputs.

Policy notes:
- Input `raw_bytes` are decoded to UTF-8 text with `errors="replace"` before tokenization.
- `infer_topk` returns per-token-position top-k ids/logprobs from prompt token logprobs, sorted by descending logprob.
- Entropy policy is pooled: mean entropy across token positions per record.
- Entropy is computed from the available top-k distribution (renormalized over returned k).
"""

from __future__ import annotations

import math
from typing import Any

from .base import Teacher
from .long_context import prepare_long_context_teacher_input

try:
    from vllm import LLM, SamplingParams
except ModuleNotFoundError:  # pragma: no cover
    LLM = None
    SamplingParams = None

try:
    from transformers import AutoTokenizer
except ModuleNotFoundError:  # pragma: no cover
    AutoTokenizer = None


class VLLMCausalLMTeacher(Teacher):
    """Minimal vLLM backend for top-k teacher signal extraction."""

    def __init__(
        self,
        model_name_or_path: str,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        max_context: int = 2048,
        batch_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        trust_remote_code: bool = False,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.tensor_parallel_size = max(1, tensor_parallel_size)
        self.dtype = dtype
        self.max_context = max(1, max_context)
        self.batch_size = max(1, batch_size)
        self.gpu_memory_utilization = gpu_memory_utilization
        self.trust_remote_code = trust_remote_code

        self._llm = None
        self._tokenizer = None

    def supports_topk(self) -> bool:
        return True

    def supports_structured(self) -> bool:
        return False

    def supports_long_context(self) -> bool:
        return True

    def supports_hidden_summary(self) -> bool:
        return False

    def startup_self_check(self, requested_top_k: int | None = None) -> dict[str, Any]:
        """Run a lightweight backend startup validation before long pipeline runs."""
        if not str(self.model_name_or_path).strip():
            raise ValueError("vLLM teacher misconfiguration: model_name_or_path must be non-empty.")
        if int(self.max_context) < 1:
            raise ValueError(f"vLLM teacher misconfiguration: max_context must be >= 1, got {self.max_context}.")
        if requested_top_k is not None and int(requested_top_k) < 1:
            raise ValueError(f"vLLM teacher misconfiguration: requested top_k must be >= 1, got {requested_top_k}.")
        if LLM is None or SamplingParams is None:
            raise ModuleNotFoundError("vLLM teacher requires vllm but it is not importable.")

        try:
            self.prepare()
        except Exception as exc:  # pragma: no cover - depends on runtime model/env
            raise RuntimeError(
                f"vLLM teacher startup check failed during initialization for model '{self.model_name_or_path}': {exc}"
            ) from exc
        finally:
            self.close()

        return {
            "backend": "vllm_causal_lm",
            "model_name_or_path": self.model_name_or_path,
            "max_context": int(self.max_context),
            "requested_top_k": int(requested_top_k) if requested_top_k is not None else None,
            "ok": True,
        }

    def prepare(self) -> None:
        if LLM is None or SamplingParams is None:
            raise ModuleNotFoundError("vllm is required for VLLMCausalLMTeacher.")

        self._llm = LLM(
            model=self.model_name_or_path,
            tensor_parallel_size=self.tensor_parallel_size,
            dtype=self.dtype,
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=self.trust_remote_code,
        )
        self._tokenizer = self._llm.get_tokenizer()


    def _prepare_stage_b_record(self, record: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(record.get("window_raw_bytes"), bytes):
            return record

        prepared = prepare_long_context_teacher_input(
            window_raw_bytes=record["window_raw_bytes"],
            target_start_offset=int(record.get("target_start_offset", 0)),
            target_end_offset=int(record.get("target_end_offset", 0)),
            max_teacher_context=int(record.get("max_teacher_context", self.max_context)),
            window_policy=str(record.get("window_policy", "center_target")),
            target_region_policy=str(record.get("target_region_policy", "preserve_full")),
        )

        out = dict(record)
        out["teacher_input_bytes"] = prepared["teacher_input_bytes"]
        out["teacher_input_text"] = prepared["teacher_input_text"]
        out["target_start_offset"] = prepared["target_start_offset"]
        out["target_end_offset"] = prepared["target_end_offset"]
        out["long_context_truncation"] = prepared["truncation_metadata"]
        out["raw_bytes"] = prepared["teacher_input_bytes"]
        return out

    def _extract_text(self, record: dict[str, Any]) -> str:
        explicit_text = record.get("teacher_input_text")
        if isinstance(explicit_text, str):
            return explicit_text

        explicit_bytes = record.get("teacher_input_bytes")
        if isinstance(explicit_bytes, bytes):
            return explicit_bytes.decode("utf-8", errors="replace")

        raw = record.get("raw_bytes", b"")
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="replace")
        if isinstance(raw, str):
            return raw
        return str(raw)

    def _truncate_prompt(self, text: str) -> str:
        assert self._tokenizer is not None
        token_ids = self._tokenizer.encode(text)
        token_ids = token_ids[: self.max_context]
        return self._tokenizer.decode(token_ids)

    @staticmethod
    def _extract_token_logprobs(token_map: Any) -> tuple[list[int], list[float]]:
        if token_map is None:
            return [], []

        ids: list[int] = []
        lps: list[float] = []
        if isinstance(token_map, dict):
            for token_id, item in token_map.items():
                ids.append(int(token_id))
                if hasattr(item, "logprob"):
                    lps.append(float(item.logprob))
                else:
                    lps.append(float(item))

        pairs = sorted(zip(ids, lps), key=lambda x: x[1], reverse=True)
        return [p[0] for p in pairs], [p[1] for p in pairs]

    @staticmethod
    def _pooled_entropy(topk_logprobs_by_pos: list[list[float]]) -> float:
        if not topk_logprobs_by_pos:
            return 0.0

        entropies: list[float] = []
        for row in topk_logprobs_by_pos:
            if not row:
                continue
            probs = [math.exp(lp) for lp in row]
            z = sum(probs)
            if z <= 0:
                continue
            probs = [p / z for p in probs]
            entropies.append(-sum(p * math.log(max(p, 1e-12)) for p in probs))

        if not entropies:
            return 0.0
        return float(sum(entropies) / len(entropies))

    def _tokenizer_vocab_size(self) -> int | None:
        if self._tokenizer is None:
            return None
        vocab_size = getattr(self._tokenizer, "vocab_size", None)
        if isinstance(vocab_size, int) and vocab_size > 0:
            return vocab_size
        get_vocab = getattr(self._tokenizer, "get_vocab", None)
        if callable(get_vocab):
            vocab = get_vocab()
            if isinstance(vocab, dict):
                return len(vocab)
        return None

    def _validate_topk_semantics(
        self,
        *,
        top_k_ids: list[list[int]],
        top_k_logprobs: list[list[float]],
        entropy: float,
        token_length: int,
    ) -> None:
        if len(top_k_ids) != len(top_k_logprobs):
            raise RuntimeError("vLLM top-k semantics violation: id/logprob row count mismatch")

        # vLLM policy: prompt_logprobs are returned for prompt token positions where
        # token maps are available. This is typically token_length - 1 rows.
        expected_max_positions = max(int(token_length) - 1, 0)
        if len(top_k_ids) > expected_max_positions:
            raise RuntimeError(
                f"vLLM top-k semantics violation: expected <= {expected_max_positions} rows, got {len(top_k_ids)}"
            )

        vocab_size = self._tokenizer_vocab_size()
        for row_ids, row_lps in zip(top_k_ids, top_k_logprobs):
            if len(row_ids) != len(row_lps):
                raise RuntimeError("vLLM top-k semantics violation: id/logprob width mismatch")
            if any(not math.isfinite(float(lp)) for lp in row_lps):
                raise RuntimeError("vLLM top-k semantics violation: encountered non-finite logprob")
            if any(float(row_lps[j]) < float(row_lps[j + 1]) for j in range(len(row_lps) - 1)):
                raise RuntimeError("vLLM top-k semantics violation: top-k logprobs are not sorted descending")
            if vocab_size is not None and any((tid < 0 or tid >= vocab_size) for tid in row_ids):
                raise RuntimeError("vLLM top-k semantics violation: token id out of tokenizer vocab range")

        if (not math.isfinite(float(entropy))) or float(entropy) < 0.0:
            raise RuntimeError("vLLM top-k semantics violation: entropy must be finite and non-negative")


    def prepare_tokenizer_only(self) -> None:
        """Load tokenizer for tokenization-cost diagnostics without inference."""
        if self._tokenizer is not None:
            return

        if AutoTokenizer is not None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
            return

        if LLM is None or SamplingParams is None:
            raise ModuleNotFoundError(
                "Tokenization diagnostics require either transformers (AutoTokenizer) or vllm to be installed."
            )

        # Fallback when only vLLM is available; this is heavier than tokenizer-only.
        self.prepare()

    def token_lengths(self, texts: list[str]) -> list[int]:
        """Return token counts for input texts without running teacher inference."""
        self.prepare_tokenizer_only()
        assert self._tokenizer is not None
        lengths: list[int] = []
        for text in texts:
            token_ids = self._tokenizer.encode(str(text))
            token_ids = token_ids[: self.max_context]
            lengths.append(len(token_ids))
        return lengths

    def infer_topk(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self._llm is None or self._tokenizer is None or SamplingParams is None:
            raise RuntimeError("Teacher must be prepared before inference.")

        outputs: list[dict[str, Any]] = []
        for offset in range(0, len(records), self.batch_size):
            batch = [self._prepare_stage_b_record(r) for r in records[offset : offset + self.batch_size]]
            prompts = [self._truncate_prompt(self._extract_text(r)) for r in batch]
            prompt_byte_lengths = [len(p.encode("utf-8", errors="replace")) for p in prompts]
            prompt_token_lengths = [len(self._tokenizer.encode(p)) for p in prompts]
            top_k = int(batch[0].get("top_k", 5)) if batch else 5

            params = SamplingParams(
                max_tokens=1,
                temperature=0.0,
                prompt_logprobs=max(1, top_k),
            )
            generated = self._llm.generate(prompts=prompts, sampling_params=params)

            for idx, item in enumerate(generated):
                per_pos_ids: list[list[int]] = []
                per_pos_lps: list[list[float]] = []
                for token_map in item.prompt_logprobs or []:
                    ids, lps = self._extract_token_logprobs(token_map)
                    if ids:
                        per_pos_ids.append(ids[:top_k])
                        per_pos_lps.append(lps[:top_k])

                token_length = int(prompt_token_lengths[idx]) if idx < len(prompt_token_lengths) else 0
                entropy_value = self._pooled_entropy(per_pos_lps)
                self._validate_topk_semantics(
                    top_k_ids=per_pos_ids,
                    top_k_logprobs=per_pos_lps,
                    entropy=entropy_value,
                    token_length=token_length,
                )

                outputs.append(
                    {
                        "top_k_ids": per_pos_ids,
                        "top_k_logprobs": per_pos_lps,
                        "entropy": entropy_value,
                        "teacher_input_token_length": token_length,
                        "teacher_input_byte_length": int(prompt_byte_lengths[idx]) if idx < len(prompt_byte_lengths) else None,
                    }
                )

        return outputs

    def infer_structured(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        raise NotImplementedError("Structured outputs are not implemented for vLLM teacher yet.")

    def close(self) -> None:
        self._llm = None
        self._tokenizer = None
