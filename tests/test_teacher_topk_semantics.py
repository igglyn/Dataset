import math
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest

from distill_factory.teachers.hf_causal_lm import HFCausalLMTeacher
from distill_factory.teachers.llamacpp_server import LlamaCppServerTeacher
from distill_factory.teachers.vllm_causal_lm import VLLMCausalLMTeacher
import distill_factory.teachers.vllm_causal_lm as vllm_causal_lm_module


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

    top_k_ids = [[9, 3, 2], [5, 4, 0], [0, 0, 0]]
    top_k_logprobs = [[-0.2, -0.4, -1.0], [-0.5, -0.6, -0.7], [0.0, 0.0, 0.0]]
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



def test_llamacpp_infer_topk_emits_per_token_signals_when_rows_have_enough_candidates() -> None:
    class _PerTokenLlamaCppTeacher(LlamaCppServerTeacher):
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
                                {"22": -0.2, "7": -0.5, "4": -1.0},
                                {"33": -0.3, "9": -0.6, "5": -1.2},
                            ],
                        }
                    }
                ]
            }

    teacher = _PerTokenLlamaCppTeacher(
        base_url="http://localhost:8080",
        max_context=32,
        default_top_k=3,
        emit_per_token_entropy=True,
        emit_per_token_top1_gap=True,
    )
    teacher.prepare()
    try:
        output = teacher.infer_topk([{"raw_bytes": b"hello", "top_k": 3}])[0]
    finally:
        teacher.close()

    assert "per_token_entropy" in output
    assert "per_token_top1_gap" in output
    assert len(output["per_token_entropy"]) == len(output["top_k_ids"])
    assert len(output["per_token_top1_gap"]) == len(output["top_k_ids"])
    assert all(math.isfinite(float(v)) for v in output["per_token_entropy"])
    assert all(math.isfinite(float(v)) for v in output["per_token_top1_gap"])
    assert all(float(v) >= 0.0 for v in output["per_token_top1_gap"])


def test_llamacpp_infer_topk_fails_for_per_token_top1_gap_with_single_candidate_rows() -> None:
    class _SingleCandidateLlamaCppTeacher(LlamaCppServerTeacher):
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
                                {"22": -0.2},
                                {"33": -0.3},
                            ],
                        }
                    }
                ]
            }

    teacher = _SingleCandidateLlamaCppTeacher(
        base_url="http://localhost:8080",
        max_context=32,
        default_top_k=1,
        emit_per_token_top1_gap=True,
    )
    teacher.prepare()
    try:
        with pytest.raises(RuntimeError, match="fewer than 2 candidates"):
            teacher.infer_topk([{"raw_bytes": b"hello", "top_k": 1}])
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



def test_hf_infer_topk_emits_per_token_selection_signals_when_enabled() -> None:
    torch = pytest.importorskip("torch")

    class _FakeTokenizer:
        vocab_size = 6

        def __call__(self, texts: list[str], return_tensors: str, padding: bool, truncation: bool, max_length: int):
            assert return_tensors == "pt"
            _ = (padding, truncation, max_length)
            input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
            attention_mask = torch.tensor([[1, 1, 1]], dtype=torch.long)
            return {"input_ids": input_ids, "attention_mask": attention_mask}

    class _FakeModelOut:
        def __init__(self, logits):
            self.logits = logits
            self.hidden_states = None

    class _FakeModel:
        def __init__(self):
            self.device = torch.device("cpu")

        def __call__(self, **kwargs):
            _ = kwargs
            # [batch=1, seq=3, vocab=6]
            logits = torch.tensor(
                [
                    [
                        [0.0, 3.0, 1.0, -1.0, -2.0, -3.0],
                        [0.0, 1.0, 2.0, -1.0, -2.0, -3.0],
                        [0.0, 0.5, 0.1, -1.0, -2.0, -3.0],
                    ]
                ],
                dtype=torch.float32,
            )
            return _FakeModelOut(logits=logits)

        def eval(self):
            return None

    teacher = HFCausalLMTeacher(
        model_name_or_path="dummy",
        max_context=16,
        batch_size=1,
        emit_per_token_entropy=True,
        emit_per_token_top1_gap=True,
    )
    teacher._tokenizer = _FakeTokenizer()
    teacher._model = _FakeModel()

    out = teacher.infer_topk([{"raw_bytes": b"abc", "top_k": 3}])[0]

    assert "per_token_entropy" in out
    assert "per_token_top1_gap" in out
    assert len(out["per_token_entropy"]) == len(out["top_k_ids"])
    assert len(out["per_token_top1_gap"]) == len(out["top_k_ids"])
    assert all(math.isfinite(float(v)) for v in out["per_token_entropy"])
    assert all(math.isfinite(float(v)) for v in out["per_token_top1_gap"])
    assert all(float(v) >= 0.0 for v in out["per_token_top1_gap"])
    assert any(0 in row for row in out["top_k_ids"])
    assert any(len(set(row)) == 1 for row in out["top_k_logprobs"])


def test_hf_infer_topk_per_token_top1_gap_is_zero_when_topk_is_one() -> None:
    torch = pytest.importorskip("torch")

    class _FakeTokenizer:
        vocab_size = 5

        def __call__(self, texts: list[str], return_tensors: str, padding: bool, truncation: bool, max_length: int):
            _ = (texts, return_tensors, padding, truncation, max_length)
            input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
            attention_mask = torch.tensor([[1, 1, 1]], dtype=torch.long)
            return {"input_ids": input_ids, "attention_mask": attention_mask}

    class _FakeModelOut:
        def __init__(self, logits):
            self.logits = logits
            self.hidden_states = None

    class _FakeModel:
        def __init__(self):
            self.device = torch.device("cpu")

        def __call__(self, **kwargs):
            _ = kwargs
            logits = torch.tensor(
                [[
                    [0.0, 2.0, 1.0, -1.0, -2.0],
                    [0.0, 1.5, 1.0, -1.0, -2.0],
                    [0.0, 0.5, 0.1, -1.0, -2.0],
                ]],
                dtype=torch.float32,
            )
            return _FakeModelOut(logits=logits)

        def eval(self):
            return None

    teacher = HFCausalLMTeacher(
        model_name_or_path="dummy",
        max_context=16,
        batch_size=1,
        emit_per_token_top1_gap=True,
    )
    teacher._tokenizer = _FakeTokenizer()
    teacher._model = _FakeModel()

    out = teacher.infer_topk([{"raw_bytes": b"abc", "top_k": 1}])[0]
    assert len(out["per_token_top1_gap"]) == len(out["top_k_ids"])
    assert all(float(v) == 0.0 for v in out["per_token_top1_gap"])


def test_hf_infer_topk_uses_zero_padding_token_ids() -> None:
    torch = pytest.importorskip("torch")

    class _FakeTokenizer:
        vocab_size = 7

        def __call__(self, texts: list[str], return_tensors: str, padding: bool, truncation: bool, max_length: int):
            _ = (texts, return_tensors, padding, truncation, max_length)
            # second sequence is padded at final position with non-zero id to
            # ensure teacher rewrites padded token ids to 0.
            input_ids = torch.tensor([[4, 5, 6], [3, 2, 9]], dtype=torch.long)
            attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.long)
            return {"input_ids": input_ids, "attention_mask": attention_mask}

    class _FakeModelOut:
        def __init__(self, logits):
            self.logits = logits
            self.hidden_states = None

    class _FakeModel:
        def __init__(self):
            self.device = torch.device("cpu")
            self.last_input_ids = None

        def __call__(self, **kwargs):
            self.last_input_ids = kwargs["input_ids"].detach().cpu()
            logits = torch.tensor(
                [
                    [
                        [0.0, 2.0, 1.0, -1.0, -2.0, -3.0, -4.0],
                        [0.0, 1.0, 2.0, -1.0, -2.0, -3.0, -4.0],
                        [0.0, 0.5, 0.1, -1.0, -2.0, -3.0, -4.0],
                    ],
                    [
                        [0.0, 1.5, 1.0, -1.0, -2.0, -3.0, -4.0],
                        [0.0, 1.2, 1.1, -1.0, -2.0, -3.0, -4.0],
                        [0.0, 0.2, 0.1, -1.0, -2.0, -3.0, -4.0],
                    ],
                ],
                dtype=torch.float32,
            )
            return _FakeModelOut(logits=logits)

        def eval(self):
            return None

    teacher = HFCausalLMTeacher(model_name_or_path="dummy", max_context=16, batch_size=2)
    teacher._tokenizer = _FakeTokenizer()
    teacher._model = _FakeModel()

    out = teacher.infer_topk([{"raw_bytes": b"abc", "top_k": 2}, {"raw_bytes": b"xy", "top_k": 2}])
    assert len(out) == 2
    assert teacher._model.last_input_ids is not None
    assert int(teacher._model.last_input_ids[1, 2].item()) == 0



def test_vllm_infer_topk_emits_per_token_selection_signals_when_enabled() -> None:
    class _FakeTokenizer:
        vocab_size = 32

        def encode(self, text: str):
            _ = text
            return [1, 2, 3, 4]

        def decode(self, token_ids):
            _ = token_ids
            return "trimmed"

    class _FakeGeneratedItem:
        def __init__(self, prompt_logprobs):
            self.prompt_logprobs = prompt_logprobs

    class _FakeLLM:
        def generate(self, prompts, sampling_params):
            _ = (prompts, sampling_params)
            # Simulate vLLM prompt_logprobs behavior: first row may be missing (None), and a row can have only one candidate.
            return [
                _FakeGeneratedItem(
                    prompt_logprobs=[
                        None,
                        {1: -0.1, 2: -0.3, 3: -0.5},
                        {4: -0.2},
                        {5: -0.4, 6: -0.9},
                    ]
                )
            ]

    class _SamplingParamsStub:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    teacher = VLLMCausalLMTeacher(
        model_name_or_path="dummy",
        max_context=16,
        batch_size=1,
        emit_per_token_entropy=True,
        emit_per_token_top1_gap=True,
    )
    teacher._tokenizer = _FakeTokenizer()
    teacher._llm = _FakeLLM()

    original_sampling_params = vllm_causal_lm_module.SamplingParams
    vllm_causal_lm_module.SamplingParams = _SamplingParamsStub
    try:
        out = teacher.infer_topk([{"raw_bytes": b"abc", "top_k": 3}])[0]
    finally:
        vllm_causal_lm_module.SamplingParams = original_sampling_params

    assert "per_token_entropy" in out
    assert "per_token_top1_gap" in out
    assert len(out["top_k_ids"]) == max(int(out["teacher_input_token_length"]) - 1, 0)
    assert len(out["per_token_entropy"]) == len(out["top_k_ids"])
    assert len(out["per_token_top1_gap"]) == len(out["top_k_ids"])
    assert all(math.isfinite(float(v)) for v in out["per_token_entropy"])
    assert all(math.isfinite(float(v)) for v in out["per_token_top1_gap"])
    assert all(float(v) >= 0.0 for v in out["per_token_top1_gap"])
    assert any(0 in row for row in out["top_k_ids"])
    assert any(len(set(row)) == 1 for row in out["top_k_logprobs"])


def test_vllm_infer_topk_per_token_top1_gap_is_zero_when_row_has_single_candidate() -> None:
    class _FakeTokenizer:
        vocab_size = 32

        def encode(self, text: str):
            _ = text
            return [1, 2, 3]

        def decode(self, token_ids):
            _ = token_ids
            return "trimmed"

    class _FakeGeneratedItem:
        def __init__(self, prompt_logprobs):
            self.prompt_logprobs = prompt_logprobs

    class _FakeLLM:
        def generate(self, prompts, sampling_params):
            _ = (prompts, sampling_params)
            return [_FakeGeneratedItem(prompt_logprobs=[None, {1: -0.25}, {2: -0.5}])]

    class _SamplingParamsStub:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    teacher = VLLMCausalLMTeacher(
        model_name_or_path="dummy",
        max_context=16,
        batch_size=1,
        emit_per_token_top1_gap=True,
    )
    teacher._tokenizer = _FakeTokenizer()
    teacher._llm = _FakeLLM()

    original_sampling_params = vllm_causal_lm_module.SamplingParams
    vllm_causal_lm_module.SamplingParams = _SamplingParamsStub
    try:
        out = teacher.infer_topk([{"raw_bytes": b"abc", "top_k": 1}])[0]
    finally:
        vllm_causal_lm_module.SamplingParams = original_sampling_params

    assert len(out["per_token_top1_gap"]) == len(out["top_k_ids"])
    assert all(float(v) == 0.0 for v in out["per_token_top1_gap"])

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
