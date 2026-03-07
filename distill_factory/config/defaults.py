"""Default config text helpers."""


def default_config_toml() -> str:
    """Return a default TOML config for the 3-stage distillation data pipeline."""
    return """[data]
input_path = "./data/raw"
file_glob = "*.txt"
encoding = "utf-8"
chunk_bytes = 4096
overlap_bytes = 256
eval_fraction = 0.1
seed = 42

[input]
preserve_document_boundaries = true
normalize_newlines = true

[output]
output_dir = "./data/processed"
format = "jsonl"
compression = "zstd"

[stage_a]
enabled = true
teacher_name = "bulk_grounding_teacher"
mode = "topk_logits"
top_k = 32
temperature = 1.0

[stage_b]
enabled = true
teacher_name = "long_context_structure_teacher"
mode = "long_context"
top_k = 32
temperature = 1.0
context_window = 8192
stride = 2048

[stage_c]
enabled = true
teacher_name = "refinement_teacher"
mode = "structured_outputs"
top_k = 16
temperature = 0.7
"""
