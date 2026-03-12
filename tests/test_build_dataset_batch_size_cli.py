from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_build_dataset_script_accepts_batch_size_in_all_stages(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "doc1.txt").write_text("hello world " * 32, encoding="utf-8")

    out_dir = tmp_path / "out"
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        f"""
[data]
input_path = "{raw_dir.as_posix()}"
file_glob = "*.txt"
encoding = "utf-8"
chunk_bytes = 96
overlap_bytes = 0
eval_fraction = 0.1
seed = 7

[input]
preserve_document_boundaries = true
normalize_newlines = true

[output]
output_dir = "{out_dir.as_posix()}"
format = "jsonl"
compression = "zstd"
dry_run = true

[stage_a]
enabled = true
teacher_name = "dummy"
backend_type = "hf"
mode = "topk_logits"
top_k = 4
temperature = 0.0
batch_size = 2

[stage_b]
enabled = true
teacher_name = "dummy"
backend_type = "hf"
mode = "topk_logits"
top_k = 4
temperature = 0.0
context_window = 128
stride = 64
batch_size = 3

[stage_c]
enabled = true
teacher_name = "dummy"
backend_type = "hf"
mode = "topk_logits"
top_k = 4
temperature = 0.0
batch_size = 5
""".strip(),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [sys.executable, "scripts/build_dataset.py", "--config", str(cfg_path)],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "PYTHONPATH": "."},
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "Training batch sizes: stage_a=2 stage_b=3 stage_c=5" in proc.stdout
    assert "Build complete:" in proc.stdout
