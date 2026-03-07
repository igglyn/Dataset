import math
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest

from distill_factory.teachers.hf_causal_lm import HFCausalLMTeacher
from distill_factory.teachers.llamacpp_server import LlamaCppServerTeacher
from distill_factory.teachers.vllm_causal_lm import VLLMCausalLMTeacher


class _TokenizerStub:
    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size


def _assert_common_topk_semantics(output: dict, vocab_size: int, expected_max_positions: int | None = None) -> None:
    ids = output["top_k_ids"]
    lps = output["top_k_logprobs"]

    assert isinstance(ids, list)
    assert isinstance(lps, list)
    assert len(ids) == len(lps)
    if expected_max_positions is not None:
        assert len(ids) <= expected_max_positions

    for row_ids, row_lps in zip(ids, lps):
        assert len(row_ids) == len(row_lps)
        assert all(0 <= int(tid) < vocab_size for tid in row_ids)
        assert all(math.isfinite(float(lp)) for lp in row_lps)
        assert all(float(row_lps[i]) >= float(row_lps[i + 1]) for i in range(len(row_lps) - 1))

    assert math.isfinite(float(output["entropy"]))
    assert float(output["entropy"]) >= 0.0


def test_hf_validation_helper_accepts_valid_semantics() -> None:
    teacher = HFCausalLMTeacher(model_name_or_path="dummy", max_context=16, batch_size=1)
    teacher._tokenizer = _TokenizerStub(vocab_size=32)

    top_k_ids = [[3, 2, 1], [7, 4, 0]]
    top_k_logprobs = [[-0.1, -0.3, -0.5], [-0.2, -0.4, -0.9]]
    teacher._validate_topk_semantics(top_k_ids=top_k_ids, top_k_logprobs=top_k_logprobs, entropy=0.25, token_length=3)


def test_hf_validation_helper_rejects_bad_semantics() -> None:
    teacher = HFCausalLMTeacher(model_name_or_path="dummy", max_context=16, batch_size=1)
    teacher._tokenizer = _TokenizerStub(vocab_size=8)

    with pytest.raises(RuntimeError, match="id/logprob row count mismatch"):
        teacher._validate_topk_semantics(
            top_k_ids=[[1, 2]],
            top_k_logprobs=[[ -0.1, -0.2], [-0.3, -0.4]],
            entropy=0.3,
            token_length=2,
        )

    with pytest.raises(RuntimeError, match="token id out of tokenizer vocab range"):
        teacher._validate_topk_semantics(
            top_k_ids=[[10, 1]],
            top_k_logprobs=[[-0.1, -0.2]],
            entropy=0.3,
            token_length=2,
        )

    with pytest.raises(RuntimeError, match="non-finite"):
        teacher._validate_topk_semantics(
            top_k_ids=[[1, 2]],
            top_k_logprobs=[[float("nan"), -0.2]],
            entropy=0.3,
            token_length=2,
        )


def test_vllm_validation_helper_accepts_valid_semantics() -> None:
    teacher = VLLMCausalLMTeacher(model_name_or_path="dummy", max_context=16, batch_size=1)
    teacher._tokenizer = _TokenizerStub(vocab_size=64)

    top_k_ids = [[9, 3, 2], [5, 4, 0]]
    top_k_logprobs = [[-0.2, -0.4, -1.0], [-0.5, -0.6, -0.7]]
    teacher._validate_topk_semantics(top_k_ids=top_k_ids, top_k_logprobs=top_k_logprobs, entropy=0.5, token_length=4)


def test_vllm_validation_helper_rejects_unsorted_or_invalid() -> None:
    teacher = VLLMCausalLMTeacher(model_name_or_path="dummy", max_context=16, batch_size=1)
    teacher._tokenizer = _TokenizerStub(vocab_size=16)

    with pytest.raises(RuntimeError, match="not sorted descending"):
        teacher._validate_topk_semantics(
            top_k_ids=[[1, 2, 3]],
            top_k_logprobs=[[-0.3, -0.1, -0.5]],
            entropy=0.2,
            token_length=2,
        )

    with pytest.raises(RuntimeError, match="entropy"):
        teacher._validate_topk_semantics(
            top_k_ids=[[1, 2]],
            top_k_logprobs=[[-0.1, -0.2]],
            entropy=-1.0,
            token_length=2,
        )






def test_llamacpp_infer_topk_openai_logprobs_semantics() -> None:
    class _FakeLlamaCppTeacher(LlamaCppServerTeacher):
        def startup_self_check(self, requested_top_k: int | None = None) -> dict[str, object]:
            return {"ok": True}

        def _http_json(self, method: str, endpoint: str, payload: dict[str, object] | None = None) -> tuple[int, object]:
            assert method == "POST"
            assert endpoint == "/v1/completions"
            return 200, {
                "choices": [
                    {
                        "logprobs": {
                            "prompt_token_ids": [11, 22, 33],
                            "top_logprobs": [
                                None,
                                {"22": -0.2, "7": -0.5},
                                {"33": -0.3, "9": -0.6},
                            ],
                        }
                    }
                ]
            }

    teacher = _FakeLlamaCppTeacher(base_url="http://localhost:8080", max_context=32, default_top_k=2)
    teacher.prepare()
    try:
        output = teacher.infer_topk([{"raw_bytes": b"hello", "top_k": 2}])[0]
    finally:
        teacher.close()

    _assert_common_topk_semantics(output, vocab_size=1000, expected_max_positions=2)
    assert output["teacher_input_token_length"] == 3
    assert len(output["top_k_ids"]) == 2


def test_llamacpp_infer_topk_fails_without_numeric_token_ids() -> None:
    class _BadLlamaCppTeacher(LlamaCppServerTeacher):
        def startup_self_check(self, requested_top_k: int | None = None) -> dict[str, object]:
            return {"ok": True}

        def _http_json(self, method: str, endpoint: str, payload: dict[str, object] | None = None) -> tuple[int, object]:
            return 200, {
                "choices": [
                    {
                        "logprobs": {
                            "prompt_token_ids": [1, 2],
                            "top_logprobs": [None, {" hello": -0.2, " world": -0.4}],
                        }
                    }
                ]
            }

    teacher = _BadLlamaCppTeacher(base_url="http://localhost:8080", max_context=32, default_top_k=2)
    teacher.prepare()
    try:
        with pytest.raises(RuntimeError, match="numeric token-id keys"):
            teacher.infer_topk([{"raw_bytes": b"hello", "top_k": 2}])
    finally:
        teacher.close()

def test_hf_runtime_capabilities_and_tokenizer_diagnostics() -> None:
    teacher = HFCausalLMTeacher(model_name_or_path="dummy", max_context=16, batch_size=1)
    capabilities = teacher.capabilities()
    assert capabilities.backend_type == "hf"
    assert capabilities.supports_topk is True
    assert capabilities.supports_structured is True
    assert capabilities.supports_tokenizer_diagnostics is True

    class _DiagTeacher(HFCausalLMTeacher):
        def token_lengths(self, texts: list[str]) -> list[int]:
            return [len(text) for text in texts]

    diag_teacher = _DiagTeacher(model_name_or_path="dummy", max_context=16, batch_size=1)
    records = [{"raw_bytes": b"abc"}, {"teacher_input_text": "xy"}]
    out = diag_teacher.tokenizer_diagnostics(records)
    assert out == [
        {"teacher_input_token_length": 3, "teacher_input_byte_length": 3},
        {"teacher_input_token_length": 2, "teacher_input_byte_length": 2},
    ]


def test_vllm_runtime_capabilities_and_tokenizer_diagnostics() -> None:
    teacher = VLLMCausalLMTeacher(model_name_or_path="dummy", max_context=16, batch_size=1)
    capabilities = teacher.capabilities()
    assert capabilities.backend_type == "vllm"
    assert capabilities.supports_topk is True
    assert capabilities.supports_structured is False
    assert capabilities.supports_tokenizer_diagnostics is True

    class _DiagTeacher(VLLMCausalLMTeacher):
        def token_lengths(self, texts: list[str]) -> list[int]:
            return [len(text) for text in texts]

        def _truncate_prompt(self, text: str) -> str:
            return text

    diag_teacher = _DiagTeacher(model_name_or_path="dummy", max_context=16, batch_size=1)
    records = [{"raw_bytes": b"abc"}, {"teacher_input_text": "xy"}]
    out = diag_teacher.tokenizer_diagnostics(records)
    assert out == [
        {"teacher_input_token_length": 3, "teacher_input_byte_length": 3},
        {"teacher_input_token_length": 2, "teacher_input_byte_length": 2},
    ]

def test_hf_backend_smoke_topk_semantics() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    teacher = HFCausalLMTeacher(
        model_name_or_path=os.getenv("HF_TEST_MODEL", "sshleifer/tiny-gpt2"),
        device_map="cpu",
        torch_dtype="float32",
        max_context=32,
        batch_size=1,
    )

    try:
        teacher.prepare()
    except Exception as exc:
        pytest.skip(f"HF smoke backend unavailable in this environment: {exc}")

    try:
        record = {"raw_bytes": b"hello teacher semantics", "top_k": 4}
        output = teacher.infer_topk([record])[0]

        token_length = int(output["teacher_input_token_length"])
        _assert_common_topk_semantics(output, vocab_size=int(teacher._tokenizer.vocab_size), expected_max_positions=token_length - 1)
        assert len(output["top_k_ids"]) == max(token_length - 1, 0)
        # HF entropy policy is pooled scalar entropy across token positions.
        assert isinstance(output["entropy"], float)
    finally:
        teacher.close()


def test_vllm_backend_smoke_topk_semantics() -> None:
    pytest.importorskip("vllm")

    model_name = os.getenv("VLLM_TEST_MODEL")
    if not model_name:
        pytest.skip("Set VLLM_TEST_MODEL to run vLLM smoke test in compatible GPU environments")

    teacher = VLLMCausalLMTeacher(
        model_name_or_path=model_name,
        dtype="auto",
        max_context=32,
        batch_size=1,
        tensor_parallel_size=1,
    )

    try:
        teacher.prepare()
    except Exception as exc:
        pytest.skip(f"vLLM smoke backend unavailable in this environment: {exc}")

    try:
        record = {"raw_bytes": b"hello teacher semantics", "top_k": 4}
        output = teacher.infer_topk([record])[0]

        token_length = int(output["teacher_input_token_length"])
        vocab_size = int(teacher._tokenizer.vocab_size) if hasattr(teacher._tokenizer, "vocab_size") else len(teacher._tokenizer.get_vocab())
        _assert_common_topk_semantics(output, vocab_size=vocab_size, expected_max_positions=max(token_length - 1, 0))
        # vLLM entropy policy is also pooled scalar entropy, computed from renormalized top-k mass.
        assert isinstance(output["entropy"], float)
    finally:
        teacher.close()
