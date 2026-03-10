from __future__ import annotations

from pathlib import Path

from distill_factory.config.schema import load_config
from scripts import validate_stage_a


def test_validate_stage_a_emits_full_stage_tables_for_disabled_stages(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "doc1.txt").write_text("hello world", encoding="utf-8")

    base_cfg = tmp_path / "base.toml"
    base_cfg.write_text(
        f"""
[data]
input_path = "{raw_dir.as_posix()}"
file_glob = "*.txt"
encoding = "utf-8"
chunk_bytes = 64
overlap_bytes = 0
eval_fraction = 0.0
seed = 123

[input]
preserve_document_boundaries = true
normalize_newlines = true

[output]
output_dir = "{(tmp_path / 'out').as_posix()}"
format = "jsonl"
compression = "zstd"

[stage_a]
enabled = true
teacher_name = "dummy"
backend_type = "hf"
mode = "topk_logits"
top_k = 4
temperature = 0.0

[stage_b]
enabled = false
teacher_name = "llamacpp_server"
backend_type = "llamacpp_server"
mode = "long_context"
top_k = 5
temperature = 0.1
context_window = 128
stride = 64
llama_base_url = "http://127.0.0.1:7112"
llama_model_hint = "stage-b"
llama_request_timeout = 12.0

[stage_c]
enabled = false
teacher_name = "llamacpp_server"
backend_type = "llamacpp_server"
mode = "topk_logits"
top_k = 6
temperature = 0.2
llama_base_url = "http://127.0.0.1:7113"
llama_model_hint = "stage-c"
llama_request_timeout = 13.0
""".strip(),
        encoding="utf-8",
    )

    captured = {}

    def _fake_run_pipeline(path: str):
        cfg = load_config(path)
        captured["config"] = cfg
        out_dir = Path(cfg.output.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        train = out_dir / "train.jsonl"
        eval_path = out_dir / "eval.jsonl"
        train.write_text("", encoding="utf-8")
        eval_path.write_text("", encoding="utf-8")
        return {
            "doc_count": 1,
            "chunk_count": 1,
            "train_path": str(train),
            "eval_path": str(eval_path),
            "skipped_records": 0,
        }

    monkeypatch.setattr(validate_stage_a, "run_pipeline", _fake_run_pipeline)

    monkeypatch.setattr(
        "sys.argv",
        [
            "validate_stage_a.py",
            "--config",
            str(base_cfg),
            "--records",
            "1",
            "--output-root",
            str(tmp_path / "validation"),
        ],
    )

    validate_stage_a.main()

    cfg = captured["config"]
    assert cfg.stage_b.enabled is False
    assert cfg.stage_b.backend_type == "llamacpp_server"
    assert cfg.stage_b.llama_base_url == "http://127.0.0.1:7112"
    assert cfg.stage_c.enabled is False
    assert cfg.stage_c.backend_type == "llamacpp_server"
    assert cfg.stage_c.llama_base_url == "http://127.0.0.1:7113"
