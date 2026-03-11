import os
from pathlib import Path

from distill_factory.config.schema import load_config
from distill_factory.pipeline.orchestrator import _bind_stage_runtime_env


def test_bind_stage_runtime_env_uses_per_stage_llamacpp_values(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        """
[data]
input_path = "./data/raw"
file_glob = "*.txt"
encoding = "utf-8"
chunk_bytes = 128
overlap_bytes = 16
eval_fraction = 0.1
seed = 1

[input]
preserve_document_boundaries = true
normalize_newlines = true

[output]
output_dir = "./data/out"
format = "jsonl"
compression = "zstd"

[stage_a]
enabled = true
teacher_name = "llamacpp_server"
backend_type = "llamacpp_server"
mode = "topk_logits"
top_k = 7
temperature = 0.1
max_context = 1024
model_name_or_path = "distilgpt2"
device_map = "cpu"
torch_dtype = "float32"
batch_size = 3
hf_pad_token_id = 0
llama_base_url = "http://127.0.0.1:7001"
llama_model_hint = "stage-a"
llama_request_timeout = 11.0

[stage_b]
enabled = true
teacher_name = "llamacpp_server"
backend_type = "llamacpp_server"
mode = "long_context"
top_k = 9
temperature = 0.2
context_window = 2048
stride = 512
llama_base_url = "http://127.0.0.1:7002"
llama_model_hint = "stage-b"
llama_request_timeout = 22.0

[stage_c]
enabled = true
teacher_name = "llamacpp_server"
backend_type = "llamacpp_server"
mode = "topk_logits"
top_k = 11
temperature = 0.3
llama_base_url = "http://127.0.0.1:7003"
llama_model_hint = "stage-c"
llama_request_timeout = 33.0
""".strip(),
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)

    _bind_stage_runtime_env("stage_a", cfg)
    assert os.environ["DISTILL_LLAMACPP_BASE_URL"] == "http://127.0.0.1:7001"
    assert os.environ["DISTILL_LLAMACPP_MODEL_HINT"] == "stage-a"
    assert os.environ["DISTILL_LLAMACPP_REQUEST_TIMEOUT"] == "11.0"
    assert os.environ["DISTILL_LLAMACPP_TOP_K"] == "7"
    assert os.environ["DISTILL_LLAMACPP_TEMPERATURE"] == "0.1"
    assert os.environ["DISTILL_LLAMACPP_MAX_CONTEXT"] == "1024"
    assert os.environ["DISTILL_HF_MODEL_NAME_OR_PATH"] == "distilgpt2"
    assert os.environ["DISTILL_HF_DEVICE_MAP"] == "cpu"
    assert os.environ["DISTILL_HF_TORCH_DTYPE"] == "float32"
    assert os.environ["DISTILL_HF_MAX_CONTEXT"] == "1024"
    assert os.environ["DISTILL_HF_BATCH_SIZE"] == "3"
    assert os.environ["DISTILL_HF_PAD_TOKEN_ID"] == "0"

    _bind_stage_runtime_env("stage_b", cfg)
    assert os.environ["DISTILL_LLAMACPP_BASE_URL"] == "http://127.0.0.1:7002"
    assert os.environ["DISTILL_LLAMACPP_MODEL_HINT"] == "stage-b"
    assert os.environ["DISTILL_LLAMACPP_REQUEST_TIMEOUT"] == "22.0"
    assert os.environ["DISTILL_LLAMACPP_TOP_K"] == "9"
    assert os.environ["DISTILL_LLAMACPP_TEMPERATURE"] == "0.2"

    _bind_stage_runtime_env("stage_c", cfg)
    assert os.environ["DISTILL_LLAMACPP_BASE_URL"] == "http://127.0.0.1:7003"
    assert os.environ["DISTILL_LLAMACPP_MODEL_HINT"] == "stage-c"
    assert os.environ["DISTILL_LLAMACPP_REQUEST_TIMEOUT"] == "33.0"
    assert os.environ["DISTILL_LLAMACPP_TOP_K"] == "11"
    assert os.environ["DISTILL_LLAMACPP_TEMPERATURE"] == "0.3"
