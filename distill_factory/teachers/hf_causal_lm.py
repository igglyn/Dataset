"""Hugging Face causal LM teacher backend for distillation outputs.

Policy notes:
- Input `raw_bytes` are decoded to UTF-8 text with `errors="replace"` before tokenization.
- `infer_topk` returns per-token-position top-k ids/logprobs from next-token logits, sorted by descending logprob.
- Entropy policy is pooled: mean entropy across token positions for each record.
- Optional per-position outputs can emit `per_token_entropy` and `per_token_top1_gap` aligned to top-k rows.
- Hidden summary policy (optional): mean pool of final hidden states over non-padding tokens.
- `infer_structured` uses deterministic generation (`do_sample=False`) and emits schema:
  task_type, prompt_text, completion_text, teacher_metadata.
"""

from __future__ import annotations

import math
from typing import Any

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

from .base import Teacher
from .runtime_base import RuntimeCapabilities, TeacherRuntime
from .long_context import prepare_long_context_teacher_input

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ModuleNotFoundError:  # pragma: no cover
    AutoModelForCausalLM = None
    AutoTokenizer = None


_DTYPE_MAP: dict[str, object] = {}
if torch is not None:
    _DTYPE_MAP = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "auto": torch.float32,
    }


class HFCausalLMTeacher(Teacher, TeacherRuntime):
    """Minimal HF causal LM backend for teacher signal extraction."""

    _RESOURCE_CACHE: dict[tuple[str, str, str, int | None], tuple[Any, Any]] = {}

    def __init__(
        self,
        model_name_or_path: str,
        device_map: str = "auto",
        torch_dtype: str = "float16",
        max_context: int = 2048,
        batch_size: int = 1,
        extract_hidden_summary: bool = False,
        emit_per_token_entropy: bool = False,
        emit_per_token_top1_gap: bool = False,
        hf_pad_token_id: int | None = None,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.max_context = max(1, max_context)
        self.batch_size = max(1, batch_size)
        self.extract_hidden_summary = bool(extract_hidden_summary)
        self.emit_per_token_entropy = bool(emit_per_token_entropy)
        self.emit_per_token_top1_gap = bool(emit_per_token_top1_gap)
        self.hf_pad_token_id = None if hf_pad_token_id is None else int(hf_pad_token_id)

        self._tokenizer = None
        self._model = None

    def supports_topk(self) -> bool:
        return True

    def supports_structured(self) -> bool:
        return True

    def supports_long_context(self) -> bool:
        return True

    def supports_hidden_summary(self) -> bool:
        return True

    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(
            backend_type="hf",
            supports_topk=True,
            supports_structured=True,
            supports_tokenizer_diagnostics=True,
        )

    def startup_self_check(self, requested_top_k: int | None = None) -> dict[str, Any]:
        """Run a lightweight backend startup validation before long pipeline runs."""
        if not str(self.model_name_or_path).strip():
            raise ValueError("HF teacher misconfiguration: model_name_or_path must be non-empty.")
        if int(self.max_context) < 1:
            raise ValueError(f"HF teacher misconfiguration: max_context must be >= 1, got {self.max_context}.")
        if requested_top_k is not None and int(requested_top_k) < 1:
            raise ValueError(f"HF teacher misconfiguration: requested top_k must be >= 1, got {requested_top_k}.")
        if torch is None:
            raise ModuleNotFoundError("HF teacher requires torch but it is not importable.")
        if AutoTokenizer is None or AutoModelForCausalLM is None:
            raise ModuleNotFoundError("HF teacher requires transformers but it is not importable.")

        try:
            self.prepare()
        except Exception as exc:  # pragma: no cover - depends on runtime model/env
            raise RuntimeError(
                f"HF teacher startup check failed during initialization for model '{self.model_name_or_path}': {exc}"
            ) from exc
        finally:
            self.close()

        return {
            "backend": "hf_causal_lm",
            "model_name_or_path": self.model_name_or_path,
            "max_context": int(self.max_context),
            "requested_top_k": int(requested_top_k) if requested_top_k is not None else None,
            "ok": True,
        }

    def prepare(self) -> None:
        if torch is None or AutoTokenizer is None or AutoModelForCausalLM is None:
            raise ModuleNotFoundError("transformers and torch are required for HFCausalLMTeacher.")

        if self._tokenizer is not None and self._model is not None:
            return

        cache_key = (self.model_name_or_path, self.device_map, self.torch_dtype, self.hf_pad_token_id)
        cached = self._RESOURCE_CACHE.get(cache_key)
        if cached is not None:
            self._tokenizer, self._model = cached
            return

        dtype = _DTYPE_MAP.get(self.torch_dtype, torch.float32)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        if self.hf_pad_token_id is not None:
            tokenizer.pad_token_id = int(self.hf_pad_token_id)

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=dtype,
            device_map=self.device_map,
        )
        if self.hf_pad_token_id is not None and hasattr(model, "config"):
            model.config.pad_token_id = int(self.hf_pad_token_id)

        model.eval()
        self._RESOURCE_CACHE[cache_key] = (tokenizer, model)
        self._tokenizer, self._model = tokenizer, model


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
        vocab_upper_bound: int | None = None,
    ) -> None:
        # HF policy: we emit next-token distributions for each observed prompt token,
        # so expected top-k rows are `max(token_length - 1, 0)`.
        expected_positions = max(int(token_length) - 1, 0)
        if len(top_k_ids) != len(top_k_logprobs):
            raise RuntimeError("HF top-k semantics violation: id/logprob row count mismatch")
        if len(top_k_ids) != expected_positions:
            raise RuntimeError(
                f"HF top-k semantics violation: expected {expected_positions} rows, got {len(top_k_ids)}"
            )

        tokenizer_vocab_size = self._tokenizer_vocab_size()
        effective_vocab_bound = vocab_upper_bound
        if effective_vocab_bound is None:
            effective_vocab_bound = tokenizer_vocab_size

        for row_ids, row_lps in zip(top_k_ids, top_k_logprobs):
            if len(row_ids) != len(row_lps):
                raise RuntimeError("HF top-k semantics violation: id/logprob width mismatch")
            if any(not math.isfinite(float(lp)) for lp in row_lps):
                raise RuntimeError("HF top-k semantics violation: encountered non-finite logprob")
            if any(int(tid) < 0 for tid in row_ids):
                raise RuntimeError("HF top-k semantics violation: token id must be non-negative")
            if effective_vocab_bound is not None and any(int(tid) >= int(effective_vocab_bound) for tid in row_ids):
                raise RuntimeError("HF top-k semantics violation: token id out of model/logit vocab range")

        if (not math.isfinite(float(entropy))) or float(entropy) < 0.0:
            raise RuntimeError("HF top-k semantics violation: entropy must be finite and non-negative")


    def prepare_tokenizer_only(self) -> None:
        """Load tokenizer without model weights for tokenization-cost diagnostics."""
        if AutoTokenizer is None:
            raise ModuleNotFoundError("transformers is required for HFCausalLMTeacher tokenizer diagnostics.")
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
            if self.hf_pad_token_id is not None:
                self._tokenizer.pad_token_id = int(self.hf_pad_token_id)

    def token_lengths(self, texts: list[str]) -> list[int]:
        """Return token counts for input texts without running teacher inference."""
        self.prepare_tokenizer_only()
        assert self._tokenizer is not None
        lengths: list[int] = []
        for text in texts:
            token_ids = self._tokenizer.encode(str(text), truncation=True, max_length=self.max_context)
            lengths.append(len(token_ids))
        return lengths

    def tokenizer_diagnostics(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        texts = [self._extract_text(self._prepare_stage_b_record(r)) for r in records]
        lengths = self.token_lengths(texts)
        out: list[dict[str, Any]] = []
        for text, token_length in zip(texts, lengths):
            out.append(
                {
                    "teacher_input_token_length": int(token_length),
                    "teacher_input_byte_length": len(text.encode("utf-8", errors="replace")),
                }
            )
        return out

    def infer_topk(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if torch is None or self._tokenizer is None or self._model is None:
            raise RuntimeError("Teacher must be prepared before inference.")

        outputs: list[dict[str, Any]] = []
        with torch.inference_mode():
            for offset in range(0, len(records), self.batch_size):
                batch = [self._prepare_stage_b_record(r) for r in records[offset : offset + self.batch_size]]
                texts = [self._extract_text(r) for r in batch]
                input_byte_lengths = [len(t.encode("utf-8", errors="replace")) for t in texts]
                need_hidden = self.extract_hidden_summary or any(bool(r.get("extract_hidden_summary", False)) for r in batch)

                tokenized = self._tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_context,
                )

                # Backend-facing padding convention: padding token ids are not
                # surfaced externally, so force padded positions to token id 0.
                # We retain attention_mask for true sequence lengths.
                if "input_ids" in tokenized and "attention_mask" in tokenized:
                    tokenized["input_ids"] = tokenized["input_ids"] * tokenized["attention_mask"].to(tokenized["input_ids"].dtype)

                if hasattr(self._model, "device"):
                    device = self._model.device
                    tokenized = {k: v.to(device) for k, v in tokenized.items()}

                model_out = self._model(**tokenized, output_hidden_states=need_hidden)
                logits = model_out.logits[:, :-1, :]
                logprobs = torch.log_softmax(logits, dim=-1)
                probs = torch.softmax(logits, dim=-1)
                ent = -(probs * logprobs).sum(dim=-1)

                pooled_hidden: list[list[float] | None] = [None for _ in range(logprobs.shape[0])]
                if need_hidden:
                    hidden_states = model_out.hidden_states
                    if hidden_states is None:
                        raise RuntimeError("extract_hidden_summary requested but model did not return hidden states")
                    final_hidden = hidden_states[-1]
                    mask = tokenized.get("attention_mask")
                    if mask is None:
                        pooled = final_hidden.mean(dim=1)
                    else:
                        m = mask.unsqueeze(-1).to(final_hidden.dtype)
                        denom = m.sum(dim=1).clamp(min=1.0)
                        pooled = (final_hidden * m).sum(dim=1) / denom
                    pooled_hidden = pooled.detach().cpu().tolist()

                for i in range(logprobs.shape[0]):
                    k = int(batch[i].get("top_k", 5))
                    token_length = int(tokenized["attention_mask"][i].sum().item()) if "attention_mask" in tokenized else int(logprobs.shape[1] + 1)
                    effective_positions = max(token_length - 1, 0)

                    effective_logprobs = logprobs[i, :effective_positions, :]
                    effective_entropy = ent[i, :effective_positions]
                    emit_per_token_entropy = self.emit_per_token_entropy or bool(batch[i].get("emit_per_token_entropy", False))
                    emit_per_token_top1_gap = self.emit_per_token_top1_gap or bool(batch[i].get("emit_per_token_top1_gap", False))

                    top_logprobs, top_ids = torch.topk(
                        effective_logprobs,
                        k=min(k, logprobs.shape[-1]),
                        dim=-1,
                        sorted=True,
                    )

                    entropy_value = float(effective_entropy.mean().item()) if effective_positions > 0 else 0.0
                    top_k_ids = top_ids.cpu().tolist()
                    top_k_logprobs = top_logprobs.cpu().tolist()
                    self._validate_topk_semantics(
                        top_k_ids=top_k_ids,
                        top_k_logprobs=top_k_logprobs,
                        entropy=entropy_value,
                        token_length=token_length,
                        vocab_upper_bound=int(logprobs.shape[-1]),
                    )

                    out_item = {
                        "top_k_ids": top_k_ids,
                        "top_k_logprobs": top_k_logprobs,
                        "entropy": entropy_value,
                        "teacher_input_token_length": token_length,
                        "teacher_input_byte_length": int(input_byte_lengths[i]),
                    }
                    if emit_per_token_entropy:
                        out_item["per_token_entropy"] = [float(v) for v in effective_entropy.detach().cpu().tolist()]
                    if emit_per_token_top1_gap:
                        if top_logprobs.shape[1] >= 2:
                            gaps = top_logprobs[:, 0] - top_logprobs[:, 1]
                            out_item["per_token_top1_gap"] = [float(v) for v in gaps.detach().cpu().tolist()]
                        else:
                            # Convention: if top-2 is unavailable (e.g. top_k=1), emit 0.0 gap.
                            out_item["per_token_top1_gap"] = [0.0 for _ in range(effective_positions)]
                    if self.extract_hidden_summary or bool(batch[i].get("extract_hidden_summary", False)):
                        hidden_vec = pooled_hidden[i]
                        if hidden_vec is None:
                            raise RuntimeError("extract_hidden_summary requested but hidden summary was not computed")
                        out_item["hidden_summary"] = hidden_vec
                    outputs.append(out_item)

        if len(outputs) != len(records):
            raise RuntimeError(
                "HF infer_topk output count mismatch: "
                f"expected {len(records)} outputs, got {len(outputs)}."
            )
        return outputs

    def infer_structured(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if torch is None or self._tokenizer is None or self._model is None:
            raise RuntimeError("Teacher must be prepared before inference.")

        outputs: list[dict[str, Any]] = []
        with torch.inference_mode():
            for record in records:
                prompt_text = str(record.get("prompt_text", self._extract_text(record)))
                task_type = str(record.get("task_type", "refinement"))

                tokenized = self._tokenizer(
                    [prompt_text],
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_context,
                )
                if hasattr(self._model, "device"):
                    tokenized = {k: v.to(self._model.device) for k, v in tokenized.items()}

                generated = self._model.generate(
                    **tokenized,
                    max_new_tokens=32,
                    do_sample=False,
                    pad_token_id=self._tokenizer.eos_token_id,
                )

                input_len = tokenized["input_ids"].shape[1]
                new_tokens = generated[0][input_len:]
                completion_text = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

                outputs.append(
                    {
                        "task_type": task_type,
                        "prompt_text": prompt_text,
                        "completion_text": completion_text,
                        "teacher_metadata": {
                            "teacher_backend": "hf_causal_lm",
                            "model_name_or_path": self.model_name_or_path,
                            "deterministic": True,
                        },
                    }
                )

        return outputs

    def close(self) -> None:
        self._model = None
        self._tokenizer = None
