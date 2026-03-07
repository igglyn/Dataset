# distill-factory

`distill-factory` is a portable Python repository focused on **teacher-data generation** and **dataset building** for distillation workflows.

This is **not** a model-training repository.

## Pipeline stages

- **Stage A**: bulk grounding teacher
- **Stage B**: long-context structure teacher
- **Stage C**: refinement teacher

## Current status

- Repository scaffolding and lightweight data pipeline modules are included.
- Real Stage A backends exist:
  - `hf_causal_lm` (Transformers + Torch)
  - `vllm_causal_lm` (vLLM)
- vLLM is top-k distillation only in this repository (no structured output generation).
- Stage C structured generation backends are intentionally not implemented yet.

## Teacher capability checks (fail-fast)

Stage runners validate teacher capabilities before expensive generation begins.

Capability flags used by the pipeline:

- `supports_topk`
- `supports_structured`
- `supports_hidden_summary`
- `supports_long_context`

Stage requirements:

- Stage A requires `supports_topk`.
- Stage B requires `supports_topk` and `supports_long_context`.
- Stage C in `structured_outputs` mode requires `supports_structured`.
- Hidden-summary extraction requires `supports_hidden_summary` in the selected stage mode.

If a teacher does not satisfy required capabilities, the run fails early with a clear error that includes teacher name, stage, mode, and missing capabilities.

## Stage A backend selection

Choose backend via config:

```toml
[stage_a]
teacher_name = "hf_causal_lm"     # or "vllm_causal_lm"
mode = "topk_logits"
```

## Hugging Face teacher setup (Stage A)

Install runtime dependencies:

```bash
pip install transformers torch
```

HF-related stage A config fields:

- `model_name_or_path`
- `device_map`
- `torch_dtype`
- `max_context`
- `batch_size`

Policy note: `raw_bytes` are decoded as UTF-8 with replacement (`errors="replace"`) before tokenization.

## vLLM teacher setup (Stage A)

Install runtime dependency:

```bash
pip install vllm
```

vLLM-related stage A config fields:

- `model_name_or_path`
- `tensor_parallel_size`
- `dtype`
- `max_context`
- `batch_size`
- `gpu_memory_utilization`
- `trust_remote_code`

Current vLLM backend behavior mirrors HF output semantics as closely as practical:

- emits `top_k_ids` and `top_k_logprobs` per token position
- emits pooled mean entropy per record
- decodes `raw_bytes` as UTF-8 with replacement before tokenization



## Teacher startup self-checks (preflight)

Before ingestion/chunking and stage execution, the orchestrator runs a fast teacher preflight check for each enabled stage/teacher combination.

The self-check validates:

- model path/name is present
- requested `top_k` is valid (`>= 1`)
- requested context length is sensible (`max_context >= 1`)
- required backend libraries are importable (`transformers`/`torch` for HF, `vllm` for vLLM)
- backend initialization succeeds (fast `prepare`/`close` cycle)

If any check fails, the pipeline exits early with a readable stage+teacher error so misconfiguration is caught before long generation runs.

## Optional byte/token length diagnostics

You can optionally log aggregated teacher-input length diagnostics to estimate tokenization cost and long-context feasibility.

Output config fields:

- `log_token_lengths` (`true`/`false`)
- `log_byte_lengths` (`true`/`false`)

When enabled, stage metrics include aggregated fields such as:

- average teacher input byte length
- average teacher input token length
- min/max teacher input token length
- average byte-to-token ratio

These are aggregated only (not per-sample dumps), and help answer questions like:

- whether your corpus tokenizes densely/sparsely for a given tokenizer
- how close Stage B windows are to context limits
- whether a teacher backend/configuration is likely to be cost-feasible

## Teacher sanity metrics (why entropy matters)

During Stage A and Stage B, the pipeline logs simple teacher-output sanity metrics:

- average entropy
- min/max entropy
- average chunk length
- average top-k width actually populated
- number of records per teacher
- number of records per stage

Entropy is a practical quality signal for distillation:

- very low entropy everywhere can indicate overconfident/peaky teacher outputs
- very high entropy everywhere can indicate noisy or uninformative supervision
- tracking entropy ranges helps compare teacher backends/settings and catch regressions early

The dataset inspector also prints these metrics from stored records, and prints embedded stage summaries when present.

## Inspecting generated datasets

Use the inspection CLI to quickly sanity-check output JSONL files:

```bash
python scripts/inspect_dataset.py --path data/processed/train.jsonl
```

The inspector prints:

- schema summary (versions + observed fields)
- counts by stage
- counts by teacher
- average chunk length in bytes
- whether top-k fields and structured fields exist
- one sample record

You can also inspect eval output:

```bash
python scripts/inspect_dataset.py --path data/processed/eval.jsonl
```


Export a few representative records for manual inspection (with optional stage/teacher filters):

```bash
python scripts/inspect_dataset.py \
  --path data/processed/train.jsonl \
  --sample-count 5 \
  --sample-stage stage_b \
  --sample-teacher hf_causal_lm \
  --export-samples data/processed/sample_preview.txt \
  --sample-format text
```

Use `--sample-format json` to write a JSON preview file instead.

## Comparing two dataset runs

Use the comparison utility to quickly spot run-to-run differences:

```bash
python scripts/compare_datasets.py --left data/run_a --right data/run_b
```

The comparison prints terminal-friendly diffs for:

- total record count
- record count by stage
- record count by teacher
- shard count
- average chunk length
- average entropy (if present)
- manifest metadata differences
- schema version differences

This is intended as a lightweight change check (not deep statistical analysis).

## Merging dataset shards

Use the merge utility to consolidate a sharded dataset into a new stable layout:

```bash
python scripts/merge_dataset_shards.py \
  --input-dir data/processed \
  --output-dir data/processed_merged
```

Optional re-sharding with a new deterministic shard size:

```bash
python scripts/merge_dataset_shards.py \
  --input-dir data/processed \
  --output-dir data/processed_resharded \
  --max-records-per-shard 50000 \
  --shard-prefix shard
```

Behavior:

- Reads shard files in deterministic sorted order.
- Preserves record lines exactly (no JSON rewriting by default).
- Does not apply extra deduplication during merge.
- Regenerates `dataset_manifest.json` in the output directory and preserves existing manifest metadata where available.

## Stop-after-stage controls (phased generation)

You can stop pipeline execution after a specific stage without editing code.

Output config field:

- `stop_after_stage` = `null` | `"stage_a"` | `"stage_b"` | `"stage_c"`

Behavior:

- Enabled stages still execute in order.
- The run stops cleanly once the requested stage completes.
- Outputs, manifests, metrics exports, and resume state are still written for the completed portion.

This is useful for phased generation, debugging intermediate supervision, and inspecting stage-specific outputs before running later stages.

## Dry-run mode (pipeline wiring validation)

Use dry-run mode to validate end-to-end pipeline wiring before spending GPU time.

Output config fields:

- `dry_run` (`true`/`false`)
- `dry_run_max_records` (int)

Behavior in dry-run mode:

- ingestion, chunking, split logic, long-context assembly (Stage B), prompt templating (Stage C), and storage all run
- teacher inference is skipped and replaced with deterministic placeholder outputs
- records include `extra_metadata.dry_run = true` and a `dry_run_note`
- applies to Stage A, Stage B, and Stage C
- only up to `dry_run_max_records` records per split are processed to keep runs cheap

Use this mode to check dataset shape, stage routing, prompt formatting, and output writing quickly.

## Failure logging and skipped-record tracking

The pipeline writes record-level failures to `<output_dir>/record_failures.jsonl` so failures are auditable instead of silently disappearing.

Each failure entry includes at least:

- `timestamp`
- `stage_name`
- `teacher_name`
- `doc_id` (when available)
- `chunk_index` (when available)
- `error_message`

Behavior:

- **Skipped record**: one record fails stage processing, is logged, and the run continues when safe.
- **Failed run**: a fatal stage-level condition (e.g., all records fail in a stage) aborts the run with a clear error.
- **Resumed run**: run restarts from resume state checkpoints and continues writing output while preserving prior progress.

Manifest includes `skipped_record_count` so you can track skipped totals across writes.

## Resumable pipeline execution

Long-running dataset builds can resume safely after interruption.

Output config fields:

- `resume` (`true`/`false`)
- `resume_policy` (`"strict"` or `"best_effort"`)

Behavior:

- When `resume = false`, pipeline behavior is unchanged.
- When `resume = true`, the pipeline loads `<output_dir>/resume_state.json` and continues from the last safe export point.
- Strict mode fails if the saved config fingerprint differs.
- Best-effort mode allows non-critical config differences (with a warning) but still fails on critical config differences.
- Resume writes only new records and avoids duplicating already written records.

Resume state file (`resume_state.json`) includes:

- `schema_version`
- `config_fingerprint` (+ `critical_config_fingerprint`)
- `completed_stages`
- split progress with completed shard ids and record counts
- `teacher_names`
- `last_update_timestamp`

## Dataset manifest (provenance)

Each dataset output directory now includes a machine-parseable manifest file:

- `dataset_manifest.json`

The manifest is updated when JSONL outputs are written and records:

- `schema_version`
- `creation_timestamp` and `updated_timestamp`
- `config_path` or `config_snapshot` (when provided)
- `enabled_stages`
- `teacher_names`
- `shard_count`
- `total_record_count` (when known)
- `format`, `compression`, and `format_settings`

This helps make dataset generation reproducible and auditable across runs.

`inspect_dataset.py` prints manifest contents (when present):

```bash
python scripts/inspect_dataset.py --path data/processed/train.jsonl
```

## Stage C prompt templates

Stage C structured-output generation supports plain Python prompt templates (no Jinja):

- `summarize_chunk`
- `answer_question_from_chunk`
- `continue_document`
- `extract_key_points`

Stage C config fields:

- `template_name`
- `template_kwargs` (dict/table)
- `deterministic` (bool)

Example:

```toml
[stage_c]
mode = "structured_outputs"
template_name = "extract_key_points"
template_kwargs = { max_points = 6 }
deterministic = true
```

Template reference example file:

- `configs/examples/stage_c_templates.toml`

Preview a prompt before expensive generation:

```bash
python scripts/preview_prompt.py \
  --template-name summarize_chunk \
  --template-kwargs-json '{"max_words": 60}' \
  --input-text "This is a sample chunk to preview formatting."
```

The preview prints the generated `task_type` and exact `prompt_text` using the same Stage C template registry used by the pipeline.

## Teacher-quality metrics export

The pipeline supports exporting a machine-readable teacher-quality summary JSON:

- `teacher_quality_metrics.json`

Export schema includes:

- `schema_version`
- `record_count`
- `entropy_histogram` (deterministic equal-width bins)
- `average_entropy_per_stage`
- `average_entropy_per_teacher`
- `record_counts_per_stage`
- `record_counts_per_teacher`

Inspection CLI behavior:

- prints metrics export location and whether the file is present
- can print entropy histogram bins in terminal with `--show-histogram`

Example:

```bash
python scripts/inspect_dataset.py --path data/processed/train.jsonl --show-histogram
```

## Long-document evaluation split (`eval_longdoc`)

For context-sensitive evaluation, you can reserve a dedicated split of longer documents instead of relying only on random held-out docs.

Data config fields:

- `eval_longdoc_min_bytes`
- `eval_longdoc_fraction`
- `eval_split_strategy` = `"random_docs" | "prefer_long_docs"`

Behavior:

- The normal doc-level train/eval split is still produced.
- If `eval_longdoc_fraction > 0`, a deterministic subset of eval documents is reassigned to `eval_longdoc`.
- With `eval_split_strategy = "prefer_long_docs"`, candidate docs meeting `eval_longdoc_min_bytes` are ranked by length and selected first.
- With `eval_split_strategy = "random_docs"`, eligible docs are selected deterministically via seeded shuffle.

Records keep explicit `split` metadata values: `train`, `eval`, or `eval_longdoc`.

## Static curriculum / replay stage mixing

The pipeline supports a **static curriculum/replay mix** across enabled stages for final export (not dynamic online scheduling).

Data config controls:

- `replay_stage_a_fraction`
- `replay_stage_b_fraction`
- `replay_stage_c_fraction`

Optional split-specific overrides:

- `train_replay_stage_a_fraction`
- `train_replay_stage_b_fraction`
- `train_replay_stage_c_fraction`
- `eval_replay_stage_a_fraction`
- `eval_replay_stage_b_fraction`
- `eval_replay_stage_c_fraction`

Behavior:

- When multiple stages are enabled, the orchestrator keeps per-stage record snapshots.
- Final `train` and `eval` exports are built by deterministic subsampling from those stage snapshots according to configured ratios.
- `stage_name` is preserved on all records so replay composition remains auditable.
- If only one stage is enabled, output behavior is unchanged.
