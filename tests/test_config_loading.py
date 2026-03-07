from pathlib import Path

from distill_factory.config.defaults import default_config_toml
from distill_factory.config.schema import load_config


def test_config_loading(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(default_config_toml(), encoding="utf-8")

    cfg = load_config(config_path)

    assert cfg.data.input_path == "./data/raw"
    assert cfg.data.file_glob == "*.txt"
    assert cfg.data.chunk_bytes == 4096
    assert cfg.input.preserve_document_boundaries is True
    assert cfg.output.format == "jsonl"
    assert cfg.stage_a.mode == "topk_logits"
    assert cfg.stage_b.mode == "long_context"
    assert cfg.stage_c.mode == "structured_outputs"


def test_example_default_toml_parses():
    cfg = load_config(Path("configs/examples/default.toml"))
    assert cfg.output.output_dir == "./data/processed"
