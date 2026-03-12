import types

import distill_factory.teachers.hf_causal_lm as hf_module
from distill_factory.teachers.hf_causal_lm import HFCausalLMTeacher


def test_hf_offload_layers_negative_rejected() -> None:
    try:
        HFCausalLMTeacher(model_name_or_path="dummy", hf_offload_layers=-1)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "hf_offload_layers" in str(exc)


def test_resolve_device_map_offloads_trailing_layers(monkeypatch) -> None:
    teacher = HFCausalLMTeacher(model_name_or_path="dummy", device_map="auto", hf_offload_layers=2)

    torch_stub = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: True),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
    )

    class _CfgFactory:
        @staticmethod
        def from_pretrained(_name):
            return types.SimpleNamespace(num_hidden_layers=6, model_type="llama")

    monkeypatch.setattr(hf_module, "torch", torch_stub)
    monkeypatch.setattr(hf_module, "AutoConfig", _CfgFactory)

    resolved = teacher._resolve_device_map()
    assert isinstance(resolved, dict)
    assert resolved[""] == "cpu"
    assert resolved["model.layers.0"] == "cuda:0"
    assert resolved["model.layers.1"] == "cuda:0"
    assert resolved["model.layers.2"] == "cuda:0"
    assert resolved["model.layers.3"] == "cuda:0"
    assert "model.layers.4" not in resolved
    assert "model.layers.5" not in resolved
    assert resolved["model.embed_tokens"] == "cuda:0"
    assert resolved["model.norm"] == "cuda:0"
    assert resolved["lm_head"] == "cuda:0"


def test_resolve_device_map_all_layers_offloaded_means_cpu(monkeypatch) -> None:
    teacher = HFCausalLMTeacher(model_name_or_path="dummy", device_map="auto", hf_offload_layers=999)

    torch_stub = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: True),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
    )

    class _CfgFactory:
        @staticmethod
        def from_pretrained(_name):
            return types.SimpleNamespace(num_hidden_layers=6, model_type="llama")

    monkeypatch.setattr(hf_module, "torch", torch_stub)
    monkeypatch.setattr(hf_module, "AutoConfig", _CfgFactory)

    resolved = teacher._resolve_device_map()
    assert resolved == "cpu"


def test_resolve_device_map_unknown_model_type_uses_shared_fallbacks(monkeypatch) -> None:
    teacher = HFCausalLMTeacher(model_name_or_path="dummy", device_map="auto", hf_offload_layers=1)

    torch_stub = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: True),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
    )

    class _CfgFactory:
        @staticmethod
        def from_pretrained(_name):
            return types.SimpleNamespace(num_hidden_layers=4, model_type="unknown_decoder")

    monkeypatch.setattr(hf_module, "torch", torch_stub)
    monkeypatch.setattr(hf_module, "AutoConfig", _CfgFactory)

    resolved = teacher._resolve_device_map()
    assert isinstance(resolved, dict)
    assert resolved[""] == "cpu"
    assert resolved["model.layers.0"] == "cuda:0"
    assert resolved["model.layers.1"] == "cuda:0"
    assert resolved["model.layers.2"] == "cuda:0"
    assert resolved["lm_head"] == "cuda:0"
    assert resolved["transformer.wte"] == "cuda:0"
